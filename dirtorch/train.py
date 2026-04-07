import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl
import dirtorch.loss as Loss

import pickle as pkl
import hashlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import shutil
import time


def expand_descriptors(descs, db=None, alpha=0, k=0):#增强描述符，用相似度来增强
    #k是增强的个数，alpha是增强系数，alpha越大增强程度越高
    assert k >= 0 and alpha >= 0, 'k and alpha must be non-negative'
    if k == 0:
        return descs
    descs = tonumpy(descs)
    n = descs.shape[0]
    db_descs = tonumpy(db if db is not None else descs)

    sim = matmul(descs, db_descs)
    if db is None:
        sim[np.diag_indices(n)] = 0

    idx = np.argpartition(sim, int(-k), axis=1)[:, int(-k):]#部分排序，类似nth-element
    #形状为 (n, k)
    '''
    原始数组: [3 1 4 1 5 9 2 6]
    部分排序后的数组: [1 1 3 4 2 6 5 9] 使得第 3 个元素（索引为 2）及其左边的元素都小于或等于它，而右边的元素都大于或等于它。
    部分排序后的索引: [3 1 0 2 6 5 4 7]
    '''

    descs_aug = np.zeros_like(descs)#形状完全一样
    for i in range(n):
        new_q = np.vstack([db_descs[j, :] * sim[i, j]**alpha for j in idx[i]])#在列方向上堆叠，行上元素个数相同
        new_q = np.vstack([descs[i], new_q])
        new_q = np.mean(new_q, axis=0)
        descs_aug[i] = new_q / np.linalg.norm(new_q)

    return descs_aug


def extract_image_features(dataset, transforms, net, ret_imgs=False, same_size=False, flip=None,
                           desc="Extract feats...", iscuda=True, threads=8, batch_size=32):#batch_size改成了16，原来是8
    """ Extract image features for a given dataset.
        Output is 2-dimensional (B, D)
    """
    same_size=True#这是新加的，原来没有
    if not same_size:#当处理不同尺寸的图像时，将 batch_size 设置为 1 并禁用 torch.backends.cudnn.benchmark 的原因与深度学习框架中的性能优化和兼容性有关。
        batch_size = 1
        old_benchmark = torch.backends.cudnn.benchmark#cudnn.benchmark 是一个布尔值，用于启用或禁用 cuDNN 的自动调优功能。
        torch.backends.cudnn.benchmark = False

    loader = get_loader(dataset, trf_chain=transforms, preprocess=net.preprocess, iscuda=iscuda,
                        output=['img'], batch_size=batch_size, threads=threads, training=False, shuffle=False)#只要图片，得到DataLoader对象

    if hasattr(net, 'eval'):
        net.eval()#评估模式

    tocpu = (lambda x: x.cpu()) if ret_imgs == 'cpu' else (lambda x: x)

    img_feats = []#形状为 (n_img, D)
    trf_images = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(loader, desc, total=1+(len(dataset)-1)//batch_size):#每一个batch(B,2,...)
            imgs = inputs[0]#0是img，应该没有1
            for i in range(len(imgs)):
                if flip and flip.pop(0):#图像翻转
                    imgs[i] = imgs[i].flip(2)
            imgs = common.variables(inputs[:1], net.iscuda)[0]#inputs[:1]感觉这个是列表套tensor
            desc = net(imgs)#前向传播一步，得到的形状为 (B, D)
            if ret_imgs:
                trf_images.append(tocpu(imgs.detach()))#detach() 是一个用于处理张量（tensor）的方法，它的主要作用是将张量从当前的计算图中分离出来，返回一个新的张量，这个新张量与原张量共享数据，但不参与梯度计算。
            del imgs
            del inputs
            if len(desc.shape) == 1:#应该是针对非批次数据，补上第一个维度为1，相当于batch_size=1
                desc.unsqueeze_(0)#esc.unsqueeze_(0) 将 esc 的形状从 (3, 64, 64) 改变为 (1, 3, 64, 64)
            img_feats.append(desc.detach())

    img_feats = torch.cat(img_feats, dim=0)
    if len(img_feats.shape) == 1:#不知道是什么
        img_feats.unsqueeze_(0)

    if not same_size:
        torch.backends.cudnn.benchmark = old_benchmark#恢复原来的设置

    if ret_imgs:
        if same_size:
            trf_images = torch.cat(trf_images, dim=0)
        return trf_images, img_feats
    return img_feats


def eval_model(db, net, trfs, pooling='mean', gemp=3, detailed=True, whiten=None,
               aqe=None, adba=None, threads=8, batch_size=1, save_feats=None,
               load_feats=None, dbg=()):#修改了threads和batch_size，原来是16
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Evaluation...")
    query_db = db.get_query_db()

    # extract DB feats
    bdescs = []#形状为（n_trfs, n_img, D）
    qdescs = []

    if not load_feats:
        trfs_list = [trfs] if isinstance(trfs, str) else trfs

        for trfs in trfs_list:
            kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)#经过Pad和Crop的预处理，形状会调至相同
            kw['same_size'] = False
            bdescs.append(extract_image_features(db, trfs, net, desc="DB", **kw))#提取特征图，格式是（n_img, D）

            # extract query feats
            qdescs.append(bdescs[-1] if db is query_db else extract_image_features(query_db, trfs, net, desc="query", **kw))

        # pool from multiple transforms (scales)
        bdescs = F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1)#normalize是按D这个维度的
        qdescs = F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1)#变成了（n_img, D）
    else:
        bdescs = np.load(os.path.join(load_feats, 'feats.bdescs.npy'))
        if query_db is not db:
            qdescs = np.load(os.path.join(load_feats, 'feats.qdescs.npy'))
        else:
            qdescs = bdescs

    if save_feats:
        mkdir(save_feats)
        np.save(os.path.join(save_feats, 'feats.bdescs.npy'), bdescs.cpu().numpy())
        if query_db is not db:
            np.save(os.path.join(save_feats, 'feats.qdescs.npy'), qdescs.cpu().numpy())

    if whiten is not None:
        bdescs = common.whiten_features(tonumpy(bdescs), net.pca, **whiten)#
        qdescs = common.whiten_features(tonumpy(qdescs), net.pca, **whiten)#

    if adba is not None:
        bdescs = expand_descriptors(bdescs, **args.adba)
    if aqe is not None:
        qdescs = expand_descriptors(qdescs, db=bdescs, **args.aqe)

    scores = matmul(qdescs, bdescs)#分数是相似度的结果，(1,1)表示第一个待查询样本和第一个数据库样本之间的相似度

    del bdescs
    del qdescs

    res = {}

    try:
        aps = [db.eval_query_AP(q, s) for q, s in enumerate(tqdm.tqdm(scores, desc='AP'))]
        if not isinstance(aps[0], dict):
            #aps = [float(e) for e in aps]
            if detailed:
                res['APs'] = [float(ap[0]) for ap in aps]
                res['P@1s']=[float(ap[1]) for ap in aps]
            # Queries with no relevants have an AP of -1
            res['mAP'] = float(np.mean([e[0] for e in aps if e[0] >= 0]))
            res['P@1'] = float(np.mean([e[1] for e in aps]))
        else:
            modes = aps[0].keys()
            for mode in modes:
                apst = [float(e[mode]) for e in aps]
                if detailed:
                    res['APs'+'-'+mode] = apst
                # Queries with no relevants have an AP of -1
                res['mAP'+'-'+mode] = float(np.mean([e for e in apst if e >= 0]))
    except NotImplementedError:
        print(" AP not implemented!")

    try:
        tops = [db.eval_query_top(q, s) for q, s in enumerate(tqdm.tqdm(scores, desc='top1'))]
        if detailed:
            res['tops'] = tops
        for k in tops[0]:
            res['top%d' % k] = float(np.mean([top[k] for top in tops]))
    except NotImplementedError:
        pass

    return res


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)

    #创建模型，加载pretrained的预训练权重加载到net中
    #net = nets.create_model(pretrained='imagenet', **checkpoint['model_options'])#此处pretrained可选'imagenet'
    if os.path.exists(r'/kaggle/working/resnet_triplet.pt'):
        net = nets.create_model(pretrained=r'/kaggle/working/resnet_triplet.pt',arch='resnet101_rmac')#此处pretrained可选'imagenet'
    else:
        net = nets.create_model(pretrained='imagenet',arch='resnet101_rmac')#此处pretrained可选'imagenet'

    net = common.switch_model_to_cuda(net, iscuda, checkpoint)#转到cuda，支持并行gpu
#    net.load_state_dict(checkpoint['state_dict'])
#    net.preprocess = checkpoint.get('preprocess', net.preprocess)
#    if 'pca' in checkpoint:
#        net.pca = checkpoint.get('pca')#{'Landmarks_clean': PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,     svd_solver='full', tol=0.0, whiten=True)}
    return net

def cleanup_and_save(model, optimizer, output_dir='/kaggle/working', max_run_time=40):
    """
    在接近12小时限制时，删除所有非.pt文件，仅保留模型文件
    :param model: 要保存的PyTorch模型
    :param output_dir: Kaggle输出目录
    :param max_run_time: 最大运行时间（秒），建议设为11.8小时（预留12分钟缓冲）
    """
    start_time = time.time()
    
#    try:
    while True:
        print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
        
        # 1. 保存最终模型
        model_path = os.path.join(output_dir, 'resnet_triplet.pt')
        optimizer_path = os.path.join(output_dir, 'resnet_triplet_optim.pt')
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"模型已保存至 {model_path}")
        print(f"优化器已保存至 {optimizer_path}")

        # 2. 删除所有非.pt文件
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if not filename.endswith('.pt') and os.path.isfile(filepath):
                os.remove(filepath)
                print(f"已删除: {filename}")
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)  # 删除整个目录
                print(f"已删除目录: {filename}")
        print(f"清理时间: {time.time()-start_time:.2f} 秒")

        # 3. 强制终止运行
        print("清理完成，退出Notebook")
        os._exit(0)
    
    #except Exception as e:
    #    print(f"清理过程中出错: {e}")
        # 尝试最后一次保存
    #    torch.save(model.state_dict(), os.path.join(output_dir, 'emergency_save.pt'))
    #    os._exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

#    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--dataset', '-d', type=str, required=False,default='Oxford5K', help='Command to load dataset')
#    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')
#    parser.add_argument('--checkpoint', type=str, required=False,default='dirtorch/data/Resnet-101-AP-GeM.pt', help='path to weights')
    parser.add_argument('--checkpoint', type=str, required=False,default='', help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')#这表示当 --detailed 出现在命令行中时，argparse 会将对应的值设置为 True。如果 --detailed 不出现在命令行中，则默认为 False。这种类型的参数通常用于表示开关或布尔选项。
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=4, help='number of thread workers')#修改了threads
#    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    parser.add_argument('--gpu', type=int, default=-1, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
#    parser.add_argument('--whiten', type=str, default='Landmarks_clean', help='applies whitening')
    parser.add_argument('--whiten', type=str, default='', help='applies whitening')#传空字符串时应该能在任何python版本运行了

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

#    dl.download_dataset(args.dataset)
#    dataset = datasets.create('Paris6K_train')
    dataset = datasets.create('CADA2000_train')#创建数据集对象

    print(dataset.get_relevants(21))
    print(dataset.get_irrelevants(21))
    print(len(dataset.get_irrelevants(21))+len(dataset.get_relevants(21)))
    print(len(dataset))
    print("***")
#    qdb=dataset.get_query_db()
#    print(qdb.imgs[0])
#    origin=qdb.get_image(21)
#    origin=transforms.ToTensor()(origin)
#    origin=transforms.Normalize(mean=[0.446, 0.370, 0.326], std=[0.331, 0.297, 0.289])(origin)
    pos_choice=np.random.choice(dataset.get_relevants(21),2)
    anchor=dataset.get_image(pos_choice[0])
    pos=dataset.get_image(pos_choice[1])
    pos=transforms.ToTensor()(pos)
#    pos=transforms.Normalize(mean=[0.446, 0.370, 0.326], std=[0.331, 0.297, 0.289])(pos)
    anchor=transforms.ToTensor()(anchor)
#    anchor=transforms.Normalize(mean=[0.446, 0.370, 0.326], std=[0.331, 0.297, 0.289])(anchor)
    print(type(anchor))
    neg_choice=np.random.choice(dataset.get_irrelevants(21),1,replace=False)
    neg=dataset.get_image(neg_choice[0])
    neg=transforms.ToTensor()(neg)
#    neg=transforms.Normalize(mean=[0.446, 0.370, 0.326], std=[0.331, 0.297, 0.289])(neg)
    #展示anchor、pos和neg图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(anchor.permute(1, 2, 0).numpy())
    plt.title('Anchor')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(pos.permute(1, 2, 0).numpy())
    plt.title('Positive')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(neg.permute(1, 2, 0).numpy())
    plt.title('Negative')
    plt.axis('off')
#    plt.subplot(1, 4, 4)
#    plt.imshow(origin.permute(1, 2, 0).numpy())
#    plt.title('Origin')
    plt.axis('off')
    plt.show()



#    print(dataset.get_irrelevants(1))
#    print(len(dataset.get_irrelevants(1)))
#    print(len(dataset.get_irrelevants(4)))
    print("***")
    '''
    index=dataset.index
    plt.figure(figsize=(8, 6))
    plt.hist(index, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Image Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()
    '''
    print("***")
    print(dataset.get_image(1))
    print(dataset.get_img_class(1))
    print(type(dataset.get_img_class(1)))
    print("Test dataset:", dataset)
    
    
    net = load_model(args.checkpoint, args.iscuda)
    
    optim=Adam(net.parameters(),lr=0.00001)
    if os.path.exists(r'/kaggle/working/resnet_triplet_optim.pt'):
        optim.load_state_dict(torch.load(r'/kaggle/working/resnet_triplet_optim.pt'))
    criterion=nn.TripletMarginLoss()
    
#    for i in net.parameters():
#        i.requires_grad = False
#    for i in net.module.fc.parameters():#kaggle上需要改一下net.module.fc.parameters()
#        i.requires_grad = True

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None
    #训练部分的代码***
    same_size=True
    if not same_size:#当处理不同尺寸的图像时，将 batch_size 设置为 1 并禁用 torch.backends.cudnn.benchmark 的原因与深度学习框架中的性能优化和兼容性有关。
        args.batch_size = 1
        old_benchmark = torch.backends.cudnn.benchmark#cudnn.benchmark 是一个布尔值，用于启用或禁用 cuDNN 的自动调优功能。
        torch.backends.cudnn.benchmark = False
    args.batch_size = 32
    
    mAP_list=[]

#    optim=Adam(net.parameters(),lr=0.0001)
    
    Epoch_num=2
    start_time=time.time()
    Ave_loss=0
    net.train()
    for epoch in range(Epoch_num):
        if epoch==0:
            for i in net.parameters():#kaggle上需要改一下net.module.fc.parameters()
                i.requires_grad = True
            
        loader = get_loader(dataset, trf_chain=args.trfs, preprocess=net.preprocess, iscuda=args.iscuda,
                            output=['img'], batch_size=args.batch_size, threads=args.threads, training=True, shuffle=True)#只要图片，得到DataLoader对象
        device=torch.device('cuda' if args.iscuda else 'cpu')
        cnt=0
        Ave_loss=0
        for inputs in tqdm.tqdm(loader, desc="Epoch%d--Training..."%(epoch), total=1+(len(dataset)-1)//args.batch_size):
            cnt+=1
        #    print(type(inputs))
        #    print(type(inputs[0][0]))
        #    print(type(inputs[0]))
        #    print(len(inputs[0]))
        #    print(inputs[0][0].shape)
            optim.zero_grad()
            anchor_imgs,positive_imgs,negative_imgs = common.variables2(inputs[:1], net.iscuda)[0]
    #        print(anchor_imgs.shape)
            anchor_imgs=net(anchor_imgs)#1*3*H*W->2048
            if len(anchor_imgs.shape) == 1:#不知道是什么
                anchor_imgs=torch.unsqueeze(anchor_imgs,0)
            positive_imgs=net(positive_imgs)
            if len(positive_imgs.shape) == 1:
                positive_imgs=torch.unsqueeze(positive_imgs,0)
            negative_imgs=net(negative_imgs)
            if len(negative_imgs.shape) == 1:
                negative_imgs=torch.unsqueeze(negative_imgs,0)
    #        print(anchor_imgs.shape)
            loss=criterion(anchor_imgs,positive_imgs,negative_imgs)
            Ave_loss+=loss.item()
            loss.backward()
            optim.step()
            
            if cnt%50==0:
                print(anchor_imgs.device)
                print("loss:",loss.item())
            if time.time()-start_time>3600*10:
                elapsed = time.time() - start_time
                print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
                cleanup_and_save(net, optim)
        
#           if cnt==1:#查看变量是否转到gpu上
#               print(type(inputs))
#               print(type(imgs))
#               print(imgs[0].shape)
#               print(imgs.device)
#           print(imgs.shape)
#           if cnt==2:
#               break
        print("Epoch %d Average Loss: %.4f"%(epoch,Ave_loss/cnt))
    
    
    #训练部分的代码***
    # Evaluate
#    dataset = datasets.create('Paris6K')
    dataset = datasets.create('CADA2000_trainvalid')
    net.eval()
    args.detailed=False
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                     threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                     save_feats=args.save_feats, load_feats=args.load_feats)
    print(res)
    dataset = datasets.create('CADA2000')
    net.eval()
    args.detailed=False
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                     threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                     save_feats=args.save_feats, load_feats=args.load_feats)
#    print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))
    print(res)

    
    for i in range(25):
#        dataset=datasets.create('Paris6K_train')
        dataset=datasets.create('CADA2000_train')
        Epoch_num=2       
        net.train()
        for epoch in range(Epoch_num):
            loader = get_loader(dataset, trf_chain=args.trfs, preprocess=net.preprocess, iscuda=args.iscuda,
                                output=['img'], batch_size=args.batch_size, threads=args.threads, training=True, shuffle=True)#只要图片，得到DataLoader对象
            device=torch.device('cuda' if args.iscuda else 'cpu')
            Ave_loss=0
            cnt=0
            for inputs in tqdm.tqdm(loader, desc="Epoch%d--Training..."%(epoch), total=1+(len(dataset)-1)//args.batch_size):
                cnt+=1
            #    print(type(inputs))
            #    print(type(inputs[0][0]))
            #    print(type(inputs[0]))
            #    print(len(inputs[0]))
            #    print(inputs[0][0].shape)
                optim.zero_grad()
                anchor_imgs,positive_imgs,negative_imgs = common.variables2(inputs[:1], net.iscuda)[0]
        #        print(anchor_imgs.shape)
                anchor_imgs=net(anchor_imgs)#1*3*H*W->2048
                if len(anchor_imgs.shape) == 1:#不知道是什么
                    anchor_imgs=torch.unsqueeze(anchor_imgs,0)
                positive_imgs=net(positive_imgs)
                if len(positive_imgs.shape) == 1:
                    positive_imgs=torch.unsqueeze(positive_imgs,0)
                negative_imgs=net(negative_imgs)
                if len(negative_imgs.shape) == 1:
                    negative_imgs=torch.unsqueeze(negative_imgs,0)
        #        print(anchor_imgs.shape)
                loss=criterion(anchor_imgs,positive_imgs,negative_imgs)
                Ave_loss+=loss.item()
                loss.backward()
                optim.step()
                if cnt%50==0:
                    print(anchor_imgs.device)
                    print("loss:",loss.item())
                
                if time.time()-start_time>3600*10:
                    elapsed = time.time() - start_time
                    print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
                    cleanup_and_save(net, optim)
            
    #           if cnt==1:#查看变量是否转到gpu上
    #               print(type(inputs))
    #               print(type(imgs))
    #               print(imgs[0].shape)
    #               print(imgs.device)
    #           print(imgs.shape)
    #           if cnt==2:
    #               break
            print("Epoch %d Average Loss: %.4f"%(epoch,Ave_loss/cnt))

        #训练部分的代码***
        # Evaluate
#        dataset = datasets.create('Paris6K')
        dataset = datasets.create('CADA2000_trainvalid')
        net.eval()
        args.detailed=True
        res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                        threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                        save_feats=args.save_feats, load_feats=args.load_feats)
        print(res)
        dataset = datasets.create('CADA2000')
        net.eval()
        args.detailed=True
        res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                        threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                        save_feats=args.save_feats, load_feats=args.load_feats)
    #    print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))
        print(res)
        mAP_list.append(res['mAP'])
    
    x_ticks=[(i+1)*4 for i in range(len(mAP_list))]
    plt.figure(figsize=(10, 5))
    plt.plot(x_ticks,mAP_list)
    plt.show()

    if args.out_json:#好像没用到out_json
        # write to file
        try:
            data = json.load(open(args.out_json))
        except IOError:
            data = {}
        data[args.dataset] = res
        mkdir(args.out_json)
        open(args.out_json, 'w').write(json.dumps(data, indent=1))
        print("saved to "+args.out_json)
    
    print(x_ticks)
    print(mAP_list)

    cleanup_and_save(net, optim)
    