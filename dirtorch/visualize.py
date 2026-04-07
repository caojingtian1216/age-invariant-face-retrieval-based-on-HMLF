import sys
import os
import os.path as osp
import pdb
import math

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
from torch.utils.data import DataLoader,Dataset

import pickle as pkl
import hashlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import shutil
import time
import random
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from dirtorch.nets.inception_resnet_v1 import InceptionResnetV1
from dirtorch.nets.iresnet import iresnet50
from dirtorch.nets.ConvNeXt import get_model
from dirtorch.nets.mtl_aifr import get_mtl_model
from dirtorch.nets.Swin_Transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
from dirtorch.nets.MobileFaceNets import mobilefacenet
from dirtorch.config import cfg

'''
trf=transforms.Compose([
    transforms.Resize((224, 224)),#将图像调整为224x224大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''
trf=transforms.Compose([
    transforms.Resize(cfg.INPUT_SIZE),#将图像调整为224x224大
    transforms.RandomCrop(cfg.INPUT_SIZE),#从图像中随机裁剪出224x224的区域
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
])
def plot_map(train_map_list,eval_map_list1,eval_map_list2,eval_map_list3):
    plt.figure(figsize=(10, 5))
    plt.plot(train_map_list, label='Train mAP', marker='o')
    plt.plot(eval_map_list1, label='Eval mAP1', marker='x')
    plt.plot(eval_map_list2, label='Eval mAP2', marker='x')
    plt.plot(eval_map_list3, label='Eval mAP3', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.legend()
    plt.grid(True)
    
    #保存
    plt.savefig('/kaggle/working/map_plot.jpg')
    plt.show()
def plot_p(train_p_list,eval_p_list1,eval_p_list2,eval_p_list3):
    plt.figure(figsize=(10, 5))
    plt.plot(train_p_list, label='Train P@1', marker='o')
    plt.plot(eval_p_list1, label='Eval P@1-1', marker='x')
    plt.plot(eval_p_list2, label='Eval P@1-2', marker='x')
    plt.plot(eval_p_list3, label='Eval P@1-3', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('P@1')
    plt.title('P@1 over Epochs')
    plt.legend()
    plt.grid(True)

    #保存
    plt.savefig('/kaggle/working/p_plot.jpg')
    plt.show()
def visualize_retrieval_results(qdb_idx,query_image, retrieval_images, retrieval_scores=None, retrieval_labels=None, query_label=None):
    titles=['(2004-2006)','(2007-2009)','(2010-2012)','（训练集）']
    """
    可视化图像检索结果
    """
    # 创建画布，调整大小和布局
    fig = plt.figure(figsize=(16, 12))
    
    # 使用GridSpec创建清晰的布局：4行5列 + 查询图像单独区域
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 0.3, 1, 1], 
                         hspace=0.5, wspace=0.3)
    
    # 1. 显示查询图像（居中显示，占用中间3列）
    ax_query = fig.add_subplot(gs[0, 1:4])
    ax_query.imshow(query_image)
    
    # 构建查询图像标题
    ax_query.set_title('Query Image', fontsize=18, fontweight='bold', pad=20)
    
    # 如果有查询标签，单独添加（不加粗）
    if query_label is not None:
        ax_query.text(0.5, -0.1, f'Label: {query_label}', 
                     ha='center', va='top', fontsize=14, 
                     transform=ax_query.transAxes)
    
    ax_query.axis('off')
    
    # 2. 添加标题（跨越所有列居中）
    ax_title = fig.add_subplot(gs[1, :])
    ax_title.text(0.5, 0.5, 'Top 10 Retrieval Results'+titles[qdb_idx], 
                  ha='center', va='center', fontsize=16, fontweight='bold',
                  transform=ax_title.transAxes)
    ax_title.axis('off')
    
    # 3. 显示检索结果（2行，每行5张图）
    for i in range(min(10, len(retrieval_images))):
        row = 2 + (i // 5)  # 第2行或第3行
        col = i % 5         # 列位置0-4
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(retrieval_images[i])
        
        # 构建标题
        title_parts = [f'Top-{i+1}']
        if retrieval_scores is not None:
            title_parts.append(f'Score: {retrieval_scores[i]:.3f}')
        if retrieval_labels is not None:
            title_parts.append(f'Label: {retrieval_labels[i]}')
            
        ax.set_title('\n'.join(title_parts), fontsize=10, pad=8)
        ax.axis('off')
    
    # 4. 如果检索图像不足10张，隐藏多余的子图
    for i in range(len(retrieval_images), 10):
        row = 2 + (i // 5)
        col = i % 5
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/retrieval_results_{titles[qdb_idx]}.jpg')
    plt.show()
'''
def plot_retrieval_results(dataset, results, idx=0,epoch=-1, topk=10):
    plt.figure(figsize=(15, 6))
    for i, photo in enumerate(results[:10]):
        plt.subplot(2, 5, i+1)  # 2行5列的子图网格，i+1 是子图的编号
        plt.imshow(dataset.get_image(photo))
        plt.axis('off')  # 不显示坐标轴

    plt.suptitle(f'Epoch {epoch} Retrieval Results for Dataset {dataset.__class__.__name__} and Query {idx} (Top {topk})')
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/Epoch_{epoch}_Dataset_{dataset.__class__.__name__}_retrieval_results_query_{idx}.jpg')
    plt.show()
'''
def plot_retrieval_results(dataset, results, idx=0,epoch=-1, topk=10):
    plt.figure(figsize=(15, 6))
    for i, photo in enumerate(results[:10]):
        plt.subplot(2, 5, i+1)  # 2行5列的子图网格，i+1 是子图的编号
        plt.imshow(dataset.get_image(photo))
        plt.axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.savefig(f'/kaggle/working/Epoch_{epoch}_Dataset_{dataset.__class__.__name__}_retrieval_results_query_{idx}.jpg')
    plt.show()

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
                res['APs'] = [float(ap[0][0]) for ap in aps]
                res['P@1s']=[float(ap[0][1]) for ap in aps]
            # Queries with no relevants have an AP of -1
            res['mAP'] = float(np.mean([e[0][0] for e in aps if e[0][0] >= 0]))
            res['P@1'] = float(np.mean([e[0][1] for e in aps]))
            res['results']=[e[1] for e in aps]#画图用
            res['scores']=[e[2] for e in aps]#画图用
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
    #    print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
        
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
            if not filename.endswith('.pt') and not filename.endswith('.jpg') and os.path.isfile(filepath):
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

class PKSampler:
    """P-K采样器：每个batch包含P个类别，每个类别K个样本"""
    
    def __init__(self, dataset, p=2, k=16, batch_size=None):
        self.dataset = dataset
        self.p = p  # 每个batch的类别数
        self.k = k  # 每个类别的样本数
        self.batch_size = batch_size or (p * k)
        
        # 按类别组织数据索引（这里用的是下标减一的）
        self.class_to_indices = {i:dataset.relevants[i] for i in range(dataset.nquery) if len(dataset.relevants[i])>0}
        
        self.classes = list(self.class_to_indices.keys())
        print(f"Found {len(self.classes)} classes")
#        for cls in self.classes:
#            print(f"Class {cls}: {len(self.class_to_indices[cls])} samples")
#            if cls>30:
#                break
    
    def __iter__(self):
        while True:
            # 随机选择P个类别
            selected_classes = random.sample(self.classes, self.p)
            batch_indices = []
            
            for cls in selected_classes:
                # 从每个类别中随机选择K个样本
                cls_indices = self.class_to_indices[cls]
                #print(type(cls_indices))
                if len(cls_indices) >= self.k:
                    selected_indices = random.sample(cls_indices, self.k)
                else:
                    # 如果样本不够，就重复采样
                    selected_indices = random.choices(cls_indices, k=self.k)
                batch_indices.extend(selected_indices)
            
            yield batch_indices
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

class TripletLoss(nn.Module):
    """在线Hard Triplet Loss"""
    
    def __init__(self, margin=0.3, hard_mining=True, batch_all_weight=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.batch_all_weight = batch_all_weight#调整hard和semi-hard triplets的权重，0.5表示两者权重相等
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [N, embedding_dim] 特征向量
            labels: [N] 标签
        """
        # 计算欧几里得距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 创建标签匹配矩阵
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 排除自己与自己的比较
        eye_mask = torch.eye(labels.size(0), device=labels.device).bool()
        
        if self.hard_mining:
            return self._hard_triplet_loss(dist_matrix, labels_equal, labels_not_equal, eye_mask)
        else:
            return self._batch_all_triplet_loss(dist_matrix, labels_equal, labels_not_equal, eye_mask)
    
    def _hard_triplet_loss(self, dist_matrix, labels_equal, labels_not_equal, eye_mask):
        """Hard negative mining triplet loss"""
        # 对于每个anchor，找到最难的positive (距离最远的同类样本)
        positive_mask = labels_equal & (~eye_mask)
        hardest_positive_dist = torch.max(
            dist_matrix * positive_mask.float() + 
            (1.0 - positive_mask.float()) * (-1e10), dim=1
        )[0]
        
        # 对于每个anchor，找到最难的negative (距离最近的异类样本)
        negative_mask = labels_not_equal
        hardest_negative_dist = torch.min(
            dist_matrix * negative_mask.float() + 
            (1.0 - negative_mask.float()) * 1e10, dim=1
        )[0]
        
        # 计算triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # 只对有有效positive的anchor计算loss
        valid_anchors = torch.sum(positive_mask.float(), dim=1) > 0
        triplet_loss = triplet_loss * valid_anchors.float()
        
        return triplet_loss.mean()
    
    def _all_triplet_loss(self, dist_matrix, labels_equal, labels_not_equal, eye_mask):
        """计算所有有效triplet的loss"""
        # 获取所有有效的triplet
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        
        # 创建triplet mask
        triplet_mask = (labels_equal.unsqueeze(2) & 
                       labels_not_equal.unsqueeze(1) & 
                       (~eye_mask).unsqueeze(2))
        
        # 计算triplet loss
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        triplet_loss = F.relu(triplet_loss)
        triplet_loss = triplet_loss * triplet_mask.float()
        
        # 计算平均loss
        num_valid_triplets = torch.sum(triplet_mask.float())
        if num_valid_triplets > 0:
            return torch.sum(triplet_loss) / num_valid_triplets
        else:
            return torch.tensor(0.0, device=triplet_loss.device)
    def _batch_all_triplet_loss(self, dist_matrix, labels_equal, labels_not_equal, eye_mask):
        """Batch-all策略：结合hard和semi-hard triplets"""
        # 获取所有有效的triplet
        anchor_positive_dist = dist_matrix.unsqueeze(2)  # [N, N, 1]
        anchor_negative_dist = dist_matrix.unsqueeze(1)  # [N, 1, N]
        
        # 创建triplet mask: anchor != positive (同类), anchor != negative (异类)
        triplet_mask = (labels_equal.unsqueeze(2) & 
                       labels_not_equal.unsqueeze(1) & 
                       (~eye_mask).unsqueeze(2))
        
        # 计算基础triplet loss
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        triplet_loss = F.relu(triplet_loss)
        
        # 分类hard和semi-hard triplets
        positive_distances = anchor_positive_dist.expand_as(triplet_loss)
        negative_distances = anchor_negative_dist.expand_as(triplet_loss)
        
        # Hard triplets: d(a,p) > d(a,n) (即positive比negative更远)
        hard_mask = (positive_distances > negative_distances) & triplet_mask
        
        # Semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin
        # 即negative比positive近，但在margin范围内
        semi_hard_mask = ((positive_distances < negative_distances) & 
                         (negative_distances < positive_distances + self.margin) & 
                         triplet_mask)
        
        # 计算hard triplets loss
        hard_triplet_loss = triplet_loss * hard_mask.float()
        num_hard_triplets = torch.sum(hard_mask.float())
        
        # 计算semi-hard triplets loss
        semi_hard_triplet_loss = triplet_loss * semi_hard_mask.float()
        num_semi_hard_triplets = torch.sum(semi_hard_mask.float())
        
        # 避免除零
        if num_hard_triplets > 0:
            avg_hard_loss = torch.sum(hard_triplet_loss) / num_hard_triplets
        else:
            avg_hard_loss = torch.tensor(0.0, device=triplet_loss.device)
            
        if num_semi_hard_triplets > 0:
            avg_semi_hard_loss = torch.sum(semi_hard_triplet_loss) / num_semi_hard_triplets
        else:
            avg_semi_hard_loss = torch.tensor(0.0, device=triplet_loss.device)
        
        # 加权平均 - 如果某类triplet不存在，则只用另一类
        if num_hard_triplets > 0 and num_semi_hard_triplets > 0:
            total_loss = (self.batch_all_weight * avg_hard_loss + 
                         (1 - self.batch_all_weight) * avg_semi_hard_loss)
        elif num_hard_triplets > 0:
            total_loss = avg_hard_loss
        elif num_semi_hard_triplets > 0:
            total_loss = avg_semi_hard_loss
        else:
            # 如果没有有效triplet，使用所有triplet的平均
            total_triplets = torch.sum(triplet_mask.float())
            if total_triplets > 0:
                total_loss = torch.sum(triplet_loss * triplet_mask.float()) / total_triplets
            else:
                total_loss = torch.tensor(0.0, device=triplet_loss.device)
        
        # 可选：存储统计信息用于调试
        if hasattr(self, '_debug_stats'):
            self._debug_stats = {
                'num_hard': num_hard_triplets.item(),
                'num_semi_hard': num_semi_hard_triplets.item(),
                'hard_loss': avg_hard_loss.item(),
                'semi_hard_loss': avg_semi_hard_loss.item()
            }
        
        return total_loss

class PKDataset(Dataset):
    def __init__(self, original_dataset, sampler):
        self.original_dataset = original_dataset
        self.sampler = sampler
        self.sampler_iter = iter(sampler)
    
    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx):
        # 获取一个batch的索引
        try:
            batch_indices = next(self.sampler_iter)
        except StopIteration:
            self.sampler_iter = iter(self.sampler)
            batch_indices = next(self.sampler_iter)
        
        # 返回batch中的所有样本
        batch_data = []
        batch_labels = []
        for idx in batch_indices:
            data, label = trf(dataset.get_image(idx)),dataset.get_img_class(idx)
            batch_data.append(data)
            batch_labels.append(label)
        
        return torch.stack(batch_data), torch.tensor(batch_labels)
def create_pk_dataloader(dataset, p=2, k=16, num_workers=0):
    """创建P-K采样的数据加载器"""
    
    sampler = PKSampler(dataset, p=p, k=k)
    pk_dataset = PKDataset(dataset, sampler)
    
    return DataLoader(pk_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

class InfoNCELoss(nn.Module):
    """
    InfoNCE 损失函数，支持多种负样本采样策略。

    Args:
        temperature (float): 温度超参数，用于缩放相似度。
        mode (str): 负样本采样模式。可选:
                    'in-batch': (默认) 使用批内所有其他不同标签的样本作为负样本。
                    'memory-bank': 使用一个外部记忆库来提供大量的负样本。
                    'hnm': 在批内进行难负样本挖掘。
        num_hard_negatives (int): 在 'hnm' 模式下，选择最难的 k 个负样本。
        memory_bank_size (int): 在 'memory-bank' 模式下，记忆库的大小。
        embedding_dim (int): 在 'memory-bank' 模式下，特征向量的维度。
    """
    def __init__(self, temperature=0.1, mode='in-batch', num_hard_negatives=16,
                 memory_bank_size=4096, embedding_dim=512):# memory_bank_size=16384
        super(InfoNCELoss, self).__init__()
        
        if mode not in ['in-batch', 'memory-bank', 'hnm']:
            raise ValueError("模式必须是 'in-batch', 'memory-bank', 或 'hnm' 之一")

        self.temperature = temperature
        self.mode = mode
        self.num_hard_negatives = num_hard_negatives
        self.memory_bank_size = memory_bank_size
        self.embedding_dim = embedding_dim

        # 为 memory-bank 模式初始化 buffer
        if self.mode == 'memory-bank':
            # register_buffer 会将张量注册为模型的一部分，它会被 .to(device) 移动，但不是模型参数
            self.register_buffer("memory_bank", F.normalize(torch.randn(memory_bank_size, embedding_dim), dim=1))
            self.register_buffer("memory_labels", torch.randint(0, 1000, (memory_bank_size,)))
            self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, embeddings, labels):
        """
        计算损失。

        Args:
            embeddings (torch.Tensor): 特征向量, 形状 [N, D].
            labels (torch.Tensor): 对应的标签, 形状 [N].

        Returns:
            torch.Tensor: 计算出的损失值 (标量).
        """
        # 0. L2 归一化特征
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 1. 创建标签匹配矩阵
        # labels_equal: [N, N], (i, j)为True表示label[i] == label[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # 2. 创建正/负样本掩码
        # positive_mask: 排除自身对角线的同类样本
        positive_mask = labels_equal & ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        # negative_mask: 所有异类样本
        negative_mask = ~labels_equal

        # 3. 根据不同模式获取负样本
        if self.mode == 'memory-bank':
            # 使用记忆库作为负样本
            all_neg_embeddings = self.memory_bank.clone().detach()
            all_neg_labels = self.memory_labels.clone().detach()
            
            # 计算与记忆库中所有样本的相似度
            neg_sim = torch.matmul(embeddings, all_neg_embeddings.T)
            
            # 记忆库中也可能包含同类样本，需要排除
            memory_labels_equal = labels.unsqueeze(1) == all_neg_labels.unsqueeze(0)
            neg_sim[memory_labels_equal] = -1e9 # 将同类样本相似度设为极小值
            
            negative_logits = neg_sim
            
            # 更新记忆库
            self._update_memory_bank(embeddings, labels)

        else: # 'in-batch' 或 'hnm' 模式
            # 计算批内相似度矩阵
            sim_matrix = torch.matmul(embeddings, embeddings.T)
            
            # 将非负样本的相似度设为极小值，以便后续处理
            # 这样 negative_logits 中就只包含有效的负样本相似度
            negative_logits = sim_matrix.masked_fill(~negative_mask, -1e9)

            if self.mode == 'hnm':
                # Hard Negative Mining: 选择 top-k 最难的负样本
                # topk 会自动处理那些被 mask 掉的极小值
                negative_logits, _ = torch.topk(negative_logits, self.num_hard_negatives, dim=1)

        # 4. 计算正样本相似度
        # 对于每个 anchor，有多个正样本时，取平均或最大。这里取最大更关注难正样本。
        # sim_matrix 在 memory-bank 模式下未计算，需单独计算
        if self.mode == 'memory-bank':
            sim_matrix = torch.matmul(embeddings, embeddings.T)
            
        positive_sim = sim_matrix.masked_fill(~positive_mask, -1e9)
        # hardest_positive_logit: [N, 1]
        hardest_positive_logit, _ = torch.max(positive_sim, dim=1, keepdim=True)

        # 5. 组合 logits 并计算损失
        # logits: [N, 1 + K], K是负样本数量
        logits = torch.cat([hardest_positive_logit, negative_logits], dim=1)
        logits /= self.temperature
        
        # 标签永远是第一个 (索引0)，因为我们把正样本放在了第一列
        labels_for_loss = torch.zeros(logits.shape[0], dtype=torch.long, device=labels.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels_for_loss)
        
        return loss

    @torch.no_grad()
    def _update_memory_bank(self, embeddings, labels):
        batch_size = embeddings.size(0)
        ptr = int(self.memory_ptr)
        
        # 确保不会越界
        assert self.memory_bank_size % batch_size == 0, "记忆库大小必须是批处理大小的整数倍"
        
        # 使用指针替换记忆库中最旧的批次
        self.memory_bank[ptr : ptr + batch_size, :] = embeddings
        self.memory_labels[ptr : ptr + batch_size] = labels
        
        # 移动指针
        ptr = (ptr + batch_size) % self.memory_bank_size
        self.memory_ptr[0] = ptr

def visualize_triplets_with_model_hq(data, labels, model, dataset=None, margin=0.3, save_dir='/kaggle/working'):
    """
    使用模型提取特征后,可视化triplets (高清版本)
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        dataset: 数据集对象(可选)
        margin: triplet loss的margin参数
        save_dir: 保存图片的目录
    """
    device = data.device
    batch_size = data.size(0)
    
    # 1. 使用模型提取特征 (保持和训练时一致的归一化处理)
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        # 如果你的TripletLoss里没有normalize，这里也不要normalize
        # embeddings = F.normalize(embeddings, p=2, dim=1)  # 注释掉这行
    
    # 2. 计算距离矩阵
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # 3. 创建标签匹配矩阵
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = ~labels_equal
    eye_mask = torch.eye(labels.size(0), device=device).bool()
    
    # 4. 收集所有有效的triplets
    easy_triplets = []
    semi_hard_triplets = []
    hard_triplets = []
    
    for anchor_idx in range(batch_size):
        positive_mask = labels_equal[anchor_idx] & (~eye_mask[anchor_idx])
        positive_indices = torch.where(positive_mask)[0]
        
        negative_mask = labels_not_equal[anchor_idx]
        negative_indices = torch.where(negative_mask)[0]
        
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue
        
        for pos_idx in positive_indices:
            pos_idx = pos_idx.item()
            d_ap = dist_matrix[anchor_idx, pos_idx].item()
            
            for neg_idx in negative_indices:
                neg_idx = neg_idx.item()
                d_an = dist_matrix[anchor_idx, neg_idx].item()
                
                triplet_loss = d_ap - d_an + margin
                
                if d_an > d_ap + margin:
                    easy_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
                elif d_ap < d_an <= d_ap + margin:
                    semi_hard_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
                else:
                    hard_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
    
    # 5. 随机抽取triplets
    def sample_triplets(triplet_list, n=5):
        if len(triplet_list) > n:
            return random.sample(triplet_list, n)
        return triplet_list
    
    easy_samples = sample_triplets(easy_triplets, 5)
    semi_hard_samples = sample_triplets(semi_hard_triplets, 5)
    hard_samples = sample_triplets(hard_triplets, 5)
    
    print(f"\n总triplets数量统计:")
    print(f"Easy triplets: {len(easy_triplets)}")
    print(f"Semi-hard triplets: {len(semi_hard_triplets)}")
    print(f"Hard triplets: {len(hard_triplets)}")
    
    # 6. 改进的图像处理函数
    def process_image_hq(img_tensor):
        """高质量图像处理"""
        # 转换为numpy (保持float64精度)
        img = img_tensor.cpu().permute(1, 2, 0).double().numpy()
        
        # 反标准化 (使用double精度)
        mean = np.array(cfg.MEAN if hasattr(cfg, 'MEAN') else [0.485, 0.456, 0.406], dtype=np.float64)
        std = np.array(cfg.STD if hasattr(cfg, 'STD') else [0.229, 0.224, 0.225], dtype=np.float64)
        
        img = img * std + mean
        
        # 裁剪到[0,1]范围
        img = np.clip(img, 0, 1)
        
        return img
    
    # 7. 改进的可视化函数
    def plot_triplet_group_hq(triplet_samples, title, filename):
        if len(triplet_samples) == 0:
            print(f"警告: 没有{title}可供展示")
            return
        
        n_triplets = len(triplet_samples)
        
        # 增大图像尺寸，提高单个图像的分辨率
        fig_width = 15  # 增加宽度
        fig_height = 5 * n_triplets  # 增加高度
        fig, axes = plt.subplots(n_triplets, 3, figsize=(fig_width, fig_height))
        
        if n_triplets == 1:
            axes = axes.reshape(1, -1)
        
        for i, triplet in enumerate(triplet_samples):
            anchor_idx = triplet['anchor']
            pos_idx = triplet['positive']
            neg_idx = triplet['negative']
            
            # 高质量图像处理
            anchor_img = process_image_hq(data[anchor_idx])
            pos_img = process_image_hq(data[pos_idx])
            neg_img = process_image_hq(data[neg_idx])
            
            # 显示图像 (关闭插值以保持像素清晰)
            axes[i, 0].imshow(anchor_img, interpolation='nearest')
            axes[i, 0].set_title(f'Anchor\nLabel: {labels[anchor_idx].item()}', 
                                fontsize=12, pad=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pos_img, interpolation='nearest')
            axes[i, 1].set_title(f'Positive\nLabel: {labels[pos_idx].item()}\nd(a,p)={triplet["d_ap"]:.3f}', 
                                fontsize=12, pad=10)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(neg_img, interpolation='nearest')
            axes[i, 2].set_title(f'Negative\nLabel: {labels[neg_idx].item()}\nd(a,n)={triplet["d_an"]:.3f}', 
                                fontsize=12, pad=10)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{title} (Total: {len(triplet_samples)})', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为suptitle留出空间
        
        # 高质量保存
        save_path = os.path.join(save_dir, filename.replace('.jpg', '_hq.png'))
        plt.savefig(save_path, 
                    dpi=300,  # 提高DPI
                    bbox_inches='tight', 
                    facecolor='white',  # 设置背景色
                    edgecolor='none',   # 去除边框
                    format='png')       # 使用PNG格式避免压缩损失
        print(f"已保存高清版本: {save_path}")
        
        plt.close()
    
    # 8. 绘制三种类型的triplets
    plot_triplet_group_hq(easy_samples, 'Easy Triplets', 'easy_triplets.jpg')
    plot_triplet_group_hq(semi_hard_samples, 'Semi-hard Triplets', 'semi_hard_triplets.jpg')
    plot_triplet_group_hq(hard_samples, 'Hard Triplets', 'hard_triplets.jpg')
    
    return {
        'easy_count': len(easy_triplets),
        'semi_hard_count': len(semi_hard_triplets),
        'hard_count': len(hard_triplets)
    }


def visualize_triplets_with_model(data, labels, model, dataset=None, margin=0.3, save_dir='/kaggle/working'):
    """
    使用模型提取特征后,可视化triplets (更准确的版本)
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        dataset: 数据集对象(可选)
        margin: triplet loss的margin参数
        save_dir: 保存图片的目录
    """
    device = data.device
    batch_size = data.size(0)
    
    # 1. 使用模型提取特征
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 2. 计算距离矩阵
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # 3. 创建标签匹配矩阵
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = ~labels_equal
    eye_mask = torch.eye(labels.size(0), device=device).bool()
    
    # 4. 收集所有有效的triplets
    easy_triplets = []
    semi_hard_triplets = []
    hard_triplets = []
    
    for anchor_idx in range(batch_size):
        positive_mask = labels_equal[anchor_idx] & (~eye_mask[anchor_idx])
        positive_indices = torch.where(positive_mask)[0]
        
        negative_mask = labels_not_equal[anchor_idx]
        negative_indices = torch.where(negative_mask)[0]
        
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue
        
        for pos_idx in positive_indices:
            pos_idx = pos_idx.item()
            d_ap = dist_matrix[anchor_idx, pos_idx].item()
            
            for neg_idx in negative_indices:
                neg_idx = neg_idx.item()
                d_an = dist_matrix[anchor_idx, neg_idx].item()
                
                triplet_loss = d_ap - d_an + margin
                
                if d_an > d_ap + margin:
                    easy_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
                elif d_ap < d_an <= d_ap + margin:
                    semi_hard_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
                else:
                    hard_triplets.append({
                        'anchor': anchor_idx,
                        'positive': pos_idx,
                        'negative': neg_idx,
                        'd_ap': d_ap,
                        'd_an': d_an,
                        'loss': triplet_loss
                    })
    
    # 5. 随机抽取triplets
    def sample_triplets(triplet_list, n=10):
        if len(triplet_list) > n:
            return random.sample(triplet_list, n)
        return triplet_list
    
    easy_samples = sample_triplets(easy_triplets, 5)
    semi_hard_samples = sample_triplets(semi_hard_triplets, 5)
    hard_samples = sample_triplets(hard_triplets, 5)
    
    print(f"\n总triplets数量统计:")
    print(f"Easy triplets: {len(easy_triplets)}")
    print(f"Semi-hard triplets: {len(semi_hard_triplets)}")
    print(f"Hard triplets: {len(hard_triplets)}")
    
    # 6. 可视化
    def plot_triplet_group(triplet_samples, title, filename):
        if len(triplet_samples) == 0:
            print(f"警告: 没有{title}可供展示")
            return
        
        n_triplets = len(triplet_samples)
        # 调整布局：3列图片 + 1列文字
        fig = plt.figure(figsize=(14, 3.5 * n_triplets))
        
        # 使用GridSpec布局，左侧留空，3列图片，右侧文字区域
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_triplets, 4, figure=fig,
                     width_ratios=[3, 3, 3, 1.5],  # anchor, positive, negative, loss文字
                     hspace=0.1, wspace=0.05,
                     left=0.15, right=0.98, top=0.90, bottom=0.10)
        
        for i, triplet in enumerate(triplet_samples):
            anchor_idx = triplet['anchor']
            pos_idx = triplet['positive']
            neg_idx = triplet['negative']
            
            # 转换图像
            anchor_img = data[anchor_idx].cpu().permute(1, 2, 0).numpy()
            pos_img = data[pos_idx].cpu().permute(1, 2, 0).numpy()
            neg_img = data[neg_idx].cpu().permute(1, 2, 0).numpy()
            
            # 反标准化
            mean = np.array(cfg.MEAN if hasattr(cfg, 'MEAN') else [0.485, 0.456, 0.406])
            std = np.array(cfg.STD if hasattr(cfg, 'STD') else [0.229, 0.224, 0.225])
            
            anchor_img = np.clip(anchor_img * std + mean, 0, 1)
            pos_img = np.clip(pos_img * std + mean, 0, 1)
            neg_img = np.clip(neg_img * std + mean, 0, 1)
            
            # 显示图像（不带标题）
            ax_anchor = fig.add_subplot(gs[i, 0])
            ax_anchor.imshow(anchor_img)
            ax_anchor.axis('off')
            
            ax_pos = fig.add_subplot(gs[i, 1])
            ax_pos.imshow(pos_img)
            ax_pos.axis('off')
            
            ax_neg = fig.add_subplot(gs[i, 2])
            ax_neg.imshow(neg_img)
            ax_neg.axis('off')
            
            # 在最右侧显示loss值
            ax_text = fig.add_subplot(gs[i, 3])
            ax_text.axis('off')
            loss_text = f'Loss: {triplet["loss"]:.4f}\nd(a,p): {triplet["d_ap"]:.3f}\nd(a,n): {triplet["d_an"]:.3f}'
            ax_text.text(0.1, 0.5, loss_text, 
                        fontsize=11, 
                        verticalalignment='center',
                        family='monospace')
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
        plt.close()
    
    plot_triplet_group(easy_samples, 'Easy Triplets', 'easy_triplets.jpg')
    plot_triplet_group(semi_hard_samples, 'Semi-hard Triplets', 'semi_hard_triplets.jpg')
    plot_triplet_group(hard_samples, 'Hard Triplets', 'hard_triplets.jpg')
    
    return {
        'easy_count': len(easy_triplets),
        'semi_hard_count': len(semi_hard_triplets),
        'hard_count': len(hard_triplets)
    }
def vvisualize_tsne(index, data, labels, model, dataset, n_labels=10, n_samples_per_label=10, save_dir='/kaggle/working'):
    """
    使用t-SNE可视化不同标签的特征分布
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        dataset: 数据集对象
        n_labels: 要抽取的标签数量
        n_samples_per_label: 每个标签抽取的样本数量
        save_dir: 保存图片的目录
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = data.device
    
    # 1. 统计每个标签的样本数量
    unique_labels = torch.unique(labels)
    label_to_indices = {}
    
    for label in unique_labels:
        indices = torch.where(labels == label)[0]
        if len(indices) >= n_samples_per_label:
            label_to_indices[label.item()] = indices.cpu().numpy()
    
    # 2. 确保有足够的标签
    available_labels = list(label_to_indices.keys())
    if len(available_labels) < n_labels:
        print(f"警告: 只找到 {len(available_labels)} 个标签有至少 {n_samples_per_label} 张图片")
        n_labels = len(available_labels)
    
    if n_labels == 0:
        print("错误: 没有标签满足条件")
        return
    
    # 3. 随机选择n_labels个标签
    selected_labels = random.sample(available_labels, n_labels)
    selected_labels.sort()
    
    print(f"\n选中的标签: {selected_labels}")
    
    # 4. 收集选中标签的样本
    selected_indices = []
    label_list = []
    
    for label in selected_labels:
        # 从该标签中随机选择n_samples_per_label个样本
        indices = label_to_indices[label]
        sampled_indices = random.sample(list(indices), n_samples_per_label)
        selected_indices.extend(sampled_indices)
        label_list.extend([label] * n_samples_per_label)
    
    selected_indices = torch.tensor(selected_indices).to(device)
    
    # 5. 提取选中样本的数据
    selected_data = data[selected_indices]
    
    print(f"选中的样本总数: {len(selected_indices)}")
    print(f"每个标签的样本数: {n_samples_per_label}")
    
    # 6. 使用模型提取特征
    model.eval()
    with torch.no_grad():
        embeddings = model(selected_data)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()
    
    # 7. 使用t-SNE降维到2D
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # 8. 绘制t-SNE图
    plt.figure(figsize=(14, 10))
    
    # 为每个标签分配一个颜色
    colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    
    for idx, label in enumerate(selected_labels):
        # 找到该标签对应的点
        mask = np.array(label_list) == label
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[colors[idx]], 
                   label=f'Label {label}',
                   s=100, 
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=0.5)
    
#    plt.xlabel('t-SNE Component 1', fontsize=14)
#    plt.ylabel('t-SNE Component 2', fontsize=14)
#    plt.title(f't-SNE Visualization of {n_labels} Labels ({n_samples_per_label} samples each)', 
#              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'tsne_visualization_{index}.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存t-SNE图至: {save_path}")
    plt.close()
    
    # 9. 额外绘制一个带有标签注释的版本
    plt.figure(figsize=(14, 10))
    
    for idx, label in enumerate(selected_labels):
        mask = np.array(label_list) == label
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[colors[idx]], 
                   label=f'Label {label}',
                   s=100, 
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=0.5)
        
        # 在每个类别的中心添加标签
        center = points.mean(axis=0)
        plt.annotate(f'{label}', 
                    xy=center, 
                    fontsize=12, 
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=colors[idx], 
                             alpha=0.3,
                             edgecolor='black'))
    
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title(f't-SNE Visualization with Labels ({n_labels} labels, {n_samples_per_label} samples each)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存带注释的版本
    save_path_annotated = os.path.join(save_dir, 'tsne_visualization_annotated.jpg')
    plt.savefig(save_path_annotated, dpi=300, bbox_inches='tight')
    print(f"已保存带注释的t-SNE图至: {save_path_annotated}")
    plt.close()
    
    # 10. 返回统计信息
    return {
        'selected_labels': selected_labels,
        'n_samples_per_label': n_samples_per_label,
        'embeddings_2d': embeddings_2d,
        'label_list': label_list
    }

def visualize_tsne(index, data, labels, model, dataset, n_labels=10, n_samples_per_label=10, save_dir='/kaggle/working'):
    """
    使用t-SNE可视化不同标签的特征分布
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        dataset: 数据集对象
        n_labels: 要抽取的标签数量
        n_samples_per_label: 每个标签抽取的样本数量
        save_dir: 保存图片的目录
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = data.device
    
    # 1. 统计每个标签的样本数量
    unique_labels = torch.unique(labels)
    label_to_indices = {}
    
    for label in unique_labels:
        indices = torch.where(labels == label)[0]
        if len(indices) >= n_samples_per_label:
            label_to_indices[label.item()] = indices.cpu().numpy()
    
    # 2. 确保有足够的标签
    available_labels = list(label_to_indices.keys())
    if len(available_labels) < n_labels:
        print(f"警告: 只找到 {len(available_labels)} 个标签有至少 {n_samples_per_label} 张图片")
        n_labels = len(available_labels)
    
    if n_labels == 0:
        print("错误: 没有标签满足条件")
        return
    
    # 3. 随机选择n_labels个标签
    selected_labels = random.sample(available_labels, n_labels)
    selected_labels.sort()
    
    print(f"\n选中的标签: {selected_labels}")
    
    # 4. 收集选中标签的样本
    selected_indices = []
    label_list = []
    
    for label in selected_labels:
        # 从该标签中随机选择n_samples_per_label个样本
        indices = label_to_indices[label]
        sampled_indices = random.sample(list(indices), n_samples_per_label)
        selected_indices.extend(sampled_indices)
        label_list.extend([label] * n_samples_per_label)
    
    selected_indices = torch.tensor(selected_indices).to(device)
    
    # 5. 提取选中样本的数据
    selected_data = data[selected_indices]
    
    print(f"选中的样本总数: {len(selected_indices)}")
    print(f"每个标签的样本数: {n_samples_per_label}")
    
    # 6. 使用模型提取特征
    model.eval()
    with torch.no_grad():
        embeddings = model(selected_data)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()
    
    # 7. 使用t-SNE降维到2D
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # 8. 绘制t-SNE图
    fig = plt.figure(figsize=(14, 10))
    
    # 为每个标签分配一个颜色
    colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    
    for idx, label in enumerate(selected_labels):
        # 找到该标签对应的点
        mask = np.array(label_list) == label
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[colors[idx]], 
                   label=f'Label {label}',
                   s=100, 
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=0.5)
    
#    plt.xlabel('t-SNE Component 1', fontsize=14)
#    plt.ylabel('t-SNE Component 2', fontsize=14)
#    plt.title(f't-SNE Visualization of {n_labels} Labels ({n_samples_per_label} samples each)', 
#              fontsize=16, fontweight='bold')
    
    # 去掉x轴和y轴的刻度
    plt.xticks([])
    plt.yticks([])
    
#    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 调整布局，在底部留白
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.02, right=0.98)
    
    # 保存图片
    save_path = os.path.join(save_dir, f'tsne_visualization_{index}.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存t-SNE图至: {save_path}")
    plt.close()
    
    # 9. 额外绘制一个带有标签注释的版本
    fig = plt.figure(figsize=(14, 10))
    
    for idx, label in enumerate(selected_labels):
        mask = np.array(label_list) == label
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[colors[idx]], 
                   label=f'Label {label}',
                   s=100, 
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=0.5)
        
        # 在每个类别的中心添加标签
        center = points.mean(axis=0)
        plt.annotate(f'{label}', 
                    xy=center, 
                    fontsize=12, 
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=colors[idx], 
                             alpha=0.3,
                             edgecolor='black'))
    
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title(f't-SNE Visualization with Labels ({n_labels} labels, {n_samples_per_label} samples each)', 
              fontsize=16, fontweight='bold')
#    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 调整布局，在底部留白
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.98)
    
    # 保存带注释的版本
    save_path_annotated = os.path.join(save_dir, 'tsne_visualization_annotated.jpg')
    plt.savefig(save_path_annotated, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存带注释的t-SNE图至: {save_path_annotated}")
    plt.close()
    
    # 10. 返回统计信息
    return {
        'selected_labels': selected_labels,
        'n_samples_per_label': n_samples_per_label,
        'embeddings_2d': embeddings_2d,
        'label_list': label_list
    }
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
    #    print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
        
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
            if not filename.endswith('.pt') and not filename.endswith('.jpg') and not filename.endswith('.png') and os.path.isfile(filepath):
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
def pplot_intra_inter_class_distance_distribution(data, labels, model, save_dir='/kaggle/working', 
                                                   max_samples_per_class=50, max_pairs=10000,
                                                   feature_batch_size=64):
    """
    绘制类内距离和类间距离的分布直方图（分批次特征提取版本）
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        save_dir: 保存图片的目录
        max_samples_per_class: 每个类别最多采样的样本数（减少内存占用）
        max_pairs: 类内和类间距离各自最多计算的样本对数量
        feature_batch_size: 特征提取时的批次大小（避免显存溢出）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = data.device
    batch_size = data.size(0)
    
    print(f"\n开始计算类内和类间距离（采样+分批模式）...")
    print(f"原始总样本数: {batch_size}")
    
    # 1. 按类别分组并采样
    unique_labels = torch.unique(labels)
    sampled_indices = []
    sampled_labels = []
    
    for label in unique_labels:
        indices = torch.where(labels == label)[0]
        n_samples = len(indices)
        
        # 如果该类别样本数超过max_samples_per_class，随机采样
        if n_samples > max_samples_per_class:
            selected = indices[torch.randperm(n_samples)[:max_samples_per_class]]
        else:
            selected = indices
        
        sampled_indices.extend(selected.cpu().tolist())
        sampled_labels.extend([label.item()] * len(selected))
    
    sampled_indices = torch.tensor(sampled_indices, device=device)
    sampled_labels = torch.tensor(sampled_labels, device=device)
    
    n_sampled = len(sampled_indices)
    print(f"采样后样本数: {n_sampled} (每类最多{max_samples_per_class}个)")
    
    # 2. **分批次**提取特征（关键优化）
    model.eval()
    embeddings_list = []
    
    print(f"正在分批提取特征 (批次大小={feature_batch_size})...")
    with torch.no_grad():
        for i in range(0, n_sampled, feature_batch_size):
            batch_indices = sampled_indices[i:i+feature_batch_size]
            batch_data = data[batch_indices]
            
            # 提取特征
            batch_embeddings = model(batch_data)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # 转移到CPU并转换为numpy（释放GPU显存）
            embeddings_list.append(batch_embeddings.cpu().numpy())
            
            # 清理GPU缓存
            del batch_data, batch_embeddings
            if (i // feature_batch_size) % 10 == 0:
                torch.cuda.empty_cache()
    
    # 合并所有特征
    embeddings_np = np.concatenate(embeddings_list, axis=0)
    labels_np = sampled_labels.cpu().numpy()
    
    print(f"特征提取完成，形状: {embeddings_np.shape}")
    
    # 3. 计算距离（在CPU上进行，避免GPU显存问题）
    intra_class_distances = []
    inter_class_distances = []
    
    print("正在计算类内距离...")
    # 计算类内距离
    intra_count = 0
    for i in range(n_sampled):
        if intra_count >= max_pairs:
            break
        
        # 找到同类样本
        same_class_mask = labels_np == labels_np[i]
        same_class_indices = np.where(same_class_mask)[0]
        same_class_indices = same_class_indices[same_class_indices > i]  # 只计算上三角
        
        if len(same_class_indices) == 0:
            continue
        
        # 计算当前样本与同类样本的距离
        for j in same_class_indices:
            if intra_count >= max_pairs:
                break
            distance = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            intra_class_distances.append(distance)
            intra_count += 1
    
    print(f"类内距离样本数: {len(intra_class_distances)}")
    
    print("正在计算类间距离...")
    # 计算类间距离（采样方式）
    inter_count = 0
    attempts = 0
    max_attempts = max_pairs * 10
    
    while inter_count < max_pairs and attempts < max_attempts:
        attempts += 1
        # 随机选择两个样本
        i, j = np.random.choice(n_sampled, 2, replace=False)
        
        # 如果是不同类别，计算距离
        if labels_np[i] != labels_np[j]:
            distance = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            inter_class_distances.append(distance)
            inter_count += 1
    
    print(f"类间距离样本数: {len(inter_class_distances)}")
    
    if len(intra_class_distances) == 0 or len(inter_class_distances) == 0:
        print("警告: 类内或类间距离为空，无法绘制")
        return None
    
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)
    
    # 4. 计算统计信息
    intra_mean = np.mean(intra_class_distances)
    inter_mean = np.mean(inter_class_distances)
    intra_std = np.std(intra_class_distances)
    inter_std = np.std(inter_class_distances)
    
    print(f"\n类内距离统计:")
    print(f"  均值: {intra_mean:.4f}")
    print(f"  标准差: {intra_std:.4f}")
    print(f"  最小值: {np.min(intra_class_distances):.4f}")
    print(f"  最大值: {np.max(intra_class_distances):.4f}")
    
    print(f"\n类间距离统计:")
    print(f"  均值: {inter_mean:.4f}")
    print(f"  标准差: {inter_std:.4f}")
    print(f"  最小值: {np.min(inter_class_distances):.4f}")
    print(f"  最大值: {np.max(inter_class_distances):.4f}")
    
    mean_distance_gap = inter_mean - intra_mean
    print(f"\n类间-类内均值距离差: {mean_distance_gap:.4f}")
    
    # 5. 绘制直方图
    plt.figure(figsize=(14, 8))
    
    bins = 50
    
    # 绘制类内距离直方图
    plt.hist(intra_class_distances, bins=bins, alpha=0.6, color='blue', 
             label=f'Intra-class (μ={intra_mean:.3f}, σ={intra_std:.3f})', 
             edgecolor='black', linewidth=0.5, density=True)
    
    # 绘制类间距离直方图
    plt.hist(inter_class_distances, bins=bins, alpha=0.6, color='red', 
             label=f'Inter-class (μ={inter_mean:.3f}, σ={inter_std:.3f})', 
             edgecolor='black', linewidth=0.5, density=True)
    
    # 绘制类内距离均值线
    plt.axvline(intra_mean, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Intra-class Mean: {intra_mean:.3f}')
    
    # 绘制类间距离均值线
    plt.axvline(inter_mean, color='darkred', linestyle='--', linewidth=2, 
                label=f'Inter-class Mean: {inter_mean:.3f}')
    
    # 标注两条均值线之间的距离
    y_max = plt.ylim()[1]
    mid_point = (intra_mean + inter_mean) / 2
    plt.annotate('', xy=(inter_mean, y_max * 0.85), xytext=(intra_mean, y_max * 0.85),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_point, y_max * 0.88, f'Gap: {mean_distance_gap:.3f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('Distance', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
#    plt.title(f'Intra-class vs Inter-class Distance Distribution\n(Sampled: {len(intra_class_distances)} intra-pairs, {len(inter_class_distances)} inter-pairs)', 
#              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 保存图片
    save_path = os.path.join(save_dir, 'distance_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n已保存距离分布图至: {save_path}")
    
    plt.close()
    
    # 6. 绘制分开的子图版本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：类内距离
    ax1.hist(intra_class_distances, bins=bins, alpha=0.7, color='blue', 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(intra_mean, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Mean: {intra_mean:.3f}')
    ax1.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Intra-class Distance Distribution\n(μ={intra_mean:.3f}, σ={intra_std:.3f}, n={len(intra_class_distances)})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：类间距离
    ax2.hist(inter_class_distances, bins=bins, alpha=0.7, color='red', 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(inter_mean, color='darkred', linestyle='--', linewidth=2, 
                label=f'Mean: {inter_mean:.3f}')
    ax2.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Inter-class Distance Distribution\n(μ={inter_mean:.3f}, σ={inter_std:.3f}, n={len(inter_class_distances)})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distance Distribution Analysis (Gap: {mean_distance_gap:.3f})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path_separate = os.path.join(save_dir, 'distance_distribution_separate.png')
    plt.savefig(save_path_separate, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存分开版本至: {save_path_separate}")
    
    plt.close()
    
    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'intra_std': intra_std,
        'inter_std': inter_std,
        'mean_gap': mean_distance_gap,
        'intra_distances': intra_class_distances,
        'inter_distances': inter_class_distances,
        'n_intra_pairs': len(intra_class_distances),
        'n_inter_pairs': len(inter_class_distances)
    }

def plot_intra_inter_class_distance_distribution(data, labels, model, save_dir='/kaggle/working', 
                                                   max_samples_per_class=50, max_pairs=10000,
                                                   feature_batch_size=64):
    """
    绘制类内距离和类间距离的分布直方图（分批次特征提取版本）
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        model: 特征提取模型
        save_dir: 保存图片的目录
        max_samples_per_class: 每个类别最多采样的样本数（减少内存占用）
        max_pairs: 类内和类间距离各自最多计算的样本对数量
        feature_batch_size: 特征提取时的批次大小（避免显存溢出）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = data.device
    batch_size = data.size(0)
    
    print(f"\n开始计算类内和类间距离（采样+分批模式）...")
    print(f"原始总样本数: {batch_size}")
    
    # 1. 按类别分组并采样
    unique_labels = torch.unique(labels)
    sampled_indices = []
    sampled_labels = []
    
    for label in unique_labels:
        indices = torch.where(labels == label)[0]
        n_samples = len(indices)
        
        # 如果该类别样本数超过max_samples_per_class，随机采样
        if n_samples > max_samples_per_class:
            selected = indices[torch.randperm(n_samples)[:max_samples_per_class]]
        else:
            selected = indices
        
        sampled_indices.extend(selected.cpu().tolist())
        sampled_labels.extend([label.item()] * len(selected))
    
    sampled_indices = torch.tensor(sampled_indices, device=device)
    sampled_labels = torch.tensor(sampled_labels, device=device)
    
    n_sampled = len(sampled_indices)
    print(f"采样后样本数: {n_sampled} (每类最多{max_samples_per_class}个)")
    
    # 2. **分批次**提取特征（关键优化）
    model.eval()
    embeddings_list = []
    
    print(f"正在分批提取特征 (批次大小={feature_batch_size})...")
    with torch.no_grad():
        for i in range(0, n_sampled, feature_batch_size):
            batch_indices = sampled_indices[i:i+feature_batch_size]
            batch_data = data[batch_indices]
            
            # 提取特征
            batch_embeddings = model(batch_data)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # 转移到CPU并转换为numpy（释放GPU显存）
            embeddings_list.append(batch_embeddings.cpu().numpy())
            
            # 清理GPU缓存
            del batch_data, batch_embeddings
            if (i // feature_batch_size) % 10 == 0:
                torch.cuda.empty_cache()
    
    # 合并所有特征
    embeddings_np = np.concatenate(embeddings_list, axis=0)
    labels_np = sampled_labels.cpu().numpy()
    
    print(f"特征提取完成，形状: {embeddings_np.shape}")
    
    # 3. 计算距离（在CPU上进行，避免GPU显存问题）
    intra_class_distances = []
    inter_class_distances = []
    
    print("正在计算类内距离...")
    # 计算类内距离
    intra_count = 0
    for i in range(n_sampled):
        if intra_count >= max_pairs:
            break
        
        # 找到同类样本
        same_class_mask = labels_np == labels_np[i]
        same_class_indices = np.where(same_class_mask)[0]
        same_class_indices = same_class_indices[same_class_indices > i]  # 只计算上三角
        
        if len(same_class_indices) == 0:
            continue
        
        # 计算当前样本与同类样本的距离
        for j in same_class_indices:
            if intra_count >= max_pairs:
                break
            distance = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            intra_class_distances.append(distance)
            intra_count += 1
    
    print(f"类内距离样本数: {len(intra_class_distances)}")
    
    print("正在计算类间距离...")
    # 计算类间距离（采样方式）
    inter_count = 0
    attempts = 0
    max_attempts = max_pairs * 10
    
    while inter_count < max_pairs and attempts < max_attempts:
        attempts += 1
        # 随机选择两个样本
        i, j = np.random.choice(n_sampled, 2, replace=False)
        
        # 如果是不同类别，计算距离
        if labels_np[i] != labels_np[j]:
            distance = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            inter_class_distances.append(distance)
            inter_count += 1
    
    print(f"类间距离样本数: {len(inter_class_distances)}")
    
    if len(intra_class_distances) == 0 or len(inter_class_distances) == 0:
        print("警告: 类内或类间距离为空，无法绘制")
        return None
    
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)
    
    # 4. 计算统计信息
    intra_mean = np.mean(intra_class_distances)
    inter_mean = np.mean(inter_class_distances)
    intra_std = np.std(intra_class_distances)
    inter_std = np.std(inter_class_distances)
    
    print(f"\n类内距离统计:")
    print(f"  均值: {intra_mean:.4f}")
    print(f"  标准差: {intra_std:.4f}")
    print(f"  最小值: {np.min(intra_class_distances):.4f}")
    print(f"  最大值: {np.max(intra_class_distances):.4f}")
    
    print(f"\n类间距离统计:")
    print(f"  均值: {inter_mean:.4f}")
    print(f"  标准差: {inter_std:.4f}")
    print(f"  最小值: {np.min(inter_class_distances):.4f}")
    print(f"  最大值: {np.max(inter_class_distances):.4f}")
    
    mean_distance_gap = inter_mean - intra_mean
    print(f"\n类间-类内均值距离差: {mean_distance_gap:.4f}")
    
    # 5. 绘制直方图
    plt.figure(figsize=(14, 8))
    
    bins = 50
    
    # 绘制类内距离直方图
    plt.hist(intra_class_distances, bins=bins, alpha=0.6, color='blue', 
             label=f'Intra-class (μ={intra_mean:.3f}, σ={intra_std:.3f})', 
             edgecolor='black', linewidth=0.5, density=True)
    
    # 绘制类间距离直方图
    plt.hist(inter_class_distances, bins=bins, alpha=0.6, color='red', 
             label=f'Inter-class (μ={inter_mean:.3f}, σ={inter_std:.3f})', 
             edgecolor='black', linewidth=0.5, density=True)
    
    # 绘制类内距离均值线
    plt.axvline(intra_mean, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Intra-class Mean: {intra_mean:.3f}')
    
    # 绘制类间距离均值线
    plt.axvline(inter_mean, color='darkred', linestyle='--', linewidth=2, 
                label=f'Inter-class Mean: {inter_mean:.3f}')
    
    # 标注两条均值线之间的距离
    y_max = plt.ylim()[1]
    mid_point = (intra_mean + inter_mean) / 2
    plt.annotate('', xy=(inter_mean, y_max * 0.85), xytext=(intra_mean, y_max * 0.85),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_point, y_max * 0.88, f'Gap: {mean_distance_gap:.3f}', 
             ha='center', va='bottom', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('Distance', fontsize=20, fontweight='bold')
    plt.ylabel('Density', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
#    plt.title(f'Intra-class vs Inter-class Distance Distribution\n(Sampled: {len(intra_class_distances)} intra-pairs, {len(inter_class_distances)} inter-pairs)', 
#              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 保存图片
    save_path = os.path.join(save_dir, 'distance_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n已保存距离分布图至: {save_path}")
    
    plt.close()
    
    # 6. 绘制分开的子图版本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：类内距离
    ax1.hist(intra_class_distances, bins=bins, alpha=0.7, color='blue', 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(intra_mean, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Mean: {intra_mean:.3f}')
    ax1.set_xlabel('Distance', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title(f'Intra-class Distance Distribution\n(μ={intra_mean:.3f}, σ={intra_std:.3f}, n={len(intra_class_distances)})', 
                  fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 右图：类间距离
    ax2.hist(inter_class_distances, bins=bins, alpha=0.7, color='red', 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(inter_mean, color='darkred', linestyle='--', linewidth=2, 
                label=f'Mean: {inter_mean:.3f}')
    ax2.set_xlabel('Distance', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_title(f'Inter-class Distance Distribution\n(μ={inter_mean:.3f}, σ={inter_std:.3f}, n={len(inter_class_distances)})', 
                  fontsize=18, fontweight='bold')
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distance Distribution Analysis (Gap: {mean_distance_gap:.3f})', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path_separate = os.path.join(save_dir, 'distance_distribution_separate.png')
    plt.savefig(save_path_separate, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存分开版本至: {save_path_separate}")
    
    plt.close()
    
    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'intra_std': intra_std,
        'inter_std': inter_std,
        'mean_gap': mean_distance_gap,
        'intra_distances': intra_class_distances,
        'inter_distances': inter_class_distances,
        'n_intra_pairs': len(intra_class_distances),
        'n_inter_pairs': len(inter_class_distances)
    }

class GradCAM:
    """Grad-CAM可视化类，用于生成注意力热力图"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: 特征提取模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册hook
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_embedding=None):
        """
        生成Class Activation Map
        
        Args:
            input_image: [1, C, H, W] 输入图像
            target_embedding: 目标特征向量（用于计算梯度，如果为None则使用特征向量的L2范数）
        
        Returns:
            cam: [H, W] 热力图 (numpy数组)
        """
        self.model.eval()
        
        # 确保输入需要梯度
        if not input_image.requires_grad:
            input_image = input_image.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_image)
        
        # 检查是否成功捕获激活值和梯度
        if self.activations is None:
            raise RuntimeError("未能捕获激活值，请检查target_layer是否正确")
        
        # 计算损失（对特征向量求L2范数或与目标特征的相似度）
        if target_embedding is None:
            # 使用L2范数作为"分数"
            score = torch.norm(output, p=2, dim=1)
        else:
            # 使用与目标特征的余弦相似度
            output_norm = F.normalize(output, p=2, dim=1)
            target_norm = F.normalize(target_embedding, p=2, dim=1)
            score = torch.sum(output_norm * target_norm, dim=1)
        
        # 反向传播
        self.model.zero_grad()
        score.backward()
        
        # 检查梯度
        if self.gradients is None:
            raise RuntimeError("未能捕获梯度，请检查target_layer是否正确")
        
        # 获取梯度和激活值
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # 全局平均池化得到权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # ReLU去除负值
        
        # 转换为numpy并归一化到[0, 1]
        cam = cam.squeeze().cpu().detach().numpy()  # 确保detach()和转numpy
        
        # 防止除零
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)  # 如果全是同一值，返回全0
        
        return cam


def visualize_attention_heatmap(data, labels, models, model_names=None, sample_idx=0,
                                 save_dir='/kaggle/working', target_layer_names=None):
    """
    可视化多个模型的注意力热力图对比
    
    Args:
        data: [batch_size, C, H, W] 图像数据张量
        labels: [batch_size] 标签张量
        models: 模型列表或单个模型
        model_names: 模型名称列表（可选）
        sample_idx: 要可视化的样本索引
        save_dir: 保存图片的目录
        target_layer_names: 目标层名称列表，每个模型对应一个（可选，会自动检测）
    """
    import cv2
    from matplotlib import cm
    
    # 确保models是列表
    if not isinstance(models, list):
        models = [models]
    
    # 设置默认模型名称
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]
    elif not isinstance(model_names, list):
        model_names = [model_names]
    
    # 设置默认target_layer_names
    if target_layer_names is None:
        target_layer_names = [None] * len(models)
    elif not isinstance(target_layer_names, list):
        target_layer_names = [target_layer_names] * len(models)
    
    assert len(models) == len(model_names) == len(target_layer_names), "模型、名称和目标层数量必须一致"
    
    device = data.device
    
    # 获取单张图像和标签
    img_tensor = data[sample_idx:sample_idx+1].clone()
    label = labels[sample_idx].item()
    
    # 图像处理函数
    def process_image_for_display(img_tensor):
        """将tensor转换为可显示的numpy图像"""
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        
        # 反标准化
        mean = np.array(cfg.MEAN if hasattr(cfg, 'MEAN') else [0.485, 0.456, 0.406])
        std = np.array(cfg.STD if hasattr(cfg, 'STD') else [0.229, 0.224, 0.225])
        
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        return img
    
    def apply_heatmap(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """将热力图叠加到原图上"""
        # 确保cam是numpy数组
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().detach().numpy()
        
        # 确保cam是2D的
        if len(cam.shape) > 2:
            cam = cam.squeeze()
        
        h, w = image.shape[:2]
        
        # 调整CAM大小到原图尺寸
        cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
        
        # 应用colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 叠加
        overlayed = alpha * heatmap + (1 - alpha) * image
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    def get_target_layer(model, target_layer_name=None):
        """获取目标层（增强版，支持多种模型）"""
        if isinstance(model, torch.nn.DataParallel):
            base_model = model.module
        else:
            base_model = model
        
        # 如果指定了target_layer_name，优先使用
        if target_layer_name is not None and hasattr(base_model, target_layer_name):
            return getattr(base_model, target_layer_name)
        
        # 自动检测：根据模型类型选择合适的层
        model_class_name = base_model.__class__.__name__
        
        # iResNet系列
        if 'IResNet' in model_class_name or 'iresnet' in model_class_name.lower():
            if hasattr(base_model, 'layer4'):
                print(f"  检测到IResNet系列，使用layer4")
                return base_model.layer4
        
        # InceptionResnetV1 (FaceNet)
        elif 'InceptionResnetV1' in model_class_name:
            if hasattr(base_model, 'block8'):
                print(f"  检测到InceptionResnetV1，使用block8")
                return base_model.block8
            elif hasattr(base_model, 'repeat_3'):
                print(f"  检测到InceptionResnetV1，使用repeat_3")
                return base_model.repeat_3
        
        # MobileFaceNet
        elif 'MobileFaceNet' in model_class_name:
            if hasattr(base_model, 'conv_6_sep'):
                print(f"  检测到MobileFaceNet，使用conv_6_sep")
                return base_model.conv_6_sep
            elif hasattr(base_model, 'conv_5'):
                print(f"  检测到MobileFaceNet，使用conv_5")
                return base_model.conv_5
        
        # 通用方法：找最后一个卷积层
        print(f"  使用通用方法查找最后一个卷积层...")
        conv_layers = []
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        
        if len(conv_layers) > 0:
            layer_name, layer = conv_layers[-1]
            print(f"  找到卷积层: {layer_name}")
            return layer
        else:
            raise ValueError("无法找到卷积层，请手动指定target_layer_name")
    
    # 处理原图
    original_img = process_image_for_display(data[sample_idx])
    
    # 为每个模型生成热力图
    print(f"\n正在为 {len(models)} 个模型生成注意力热力图...")
    results = []
    
    for model, model_name, target_layer_name in zip(models, model_names, target_layer_names):
        model.eval()
        
        try:
            # 获取目标层并创建Grad-CAM
            target_layer = get_target_layer(model, target_layer_name)
            print(f"  处理 {model_name}, 目标层: {target_layer.__class__.__name__}")
            
            grad_cam = GradCAM(model, target_layer)
            
            # 生成CAM
            cam = grad_cam.generate_cam(img_tensor)
            
            print(f"  生成的CAM形状: {cam.shape}, 类型: {type(cam)}, 范围: [{cam.min():.3f}, {cam.max():.3f}]")
            
            # 叠加热力图
            overlayed_img = apply_heatmap(original_img, cam)
            
            results.append({
                'model_name': model_name,
                'overlayed': overlayed_img,
                'cam': cam
            })
            
        except Exception as e:
            print(f"  处理 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("警告: 没有成功生成任何热力图")
        return None
    
    # 可视化结果
    n_models = len(results)
    n_cols = n_models + 1  # 原图 + 各个模型的overlay图
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    # 显示原图
    axes[0].imshow(original_img)
    axes[0].set_title('Original', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 显示各个模型的overlay图
    for i, result in enumerate(results):
        axes[i+1].imshow(result['overlayed'])
        axes[i+1].set_title(f'{result["model_name"]}', 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, f'attention_comparison_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n已保存注意力对比图至: {save_path}")
    
    plt.close()
    
    return {
        'sample_idx': sample_idx,
        'label': label,
        'original': original_img,
        'results': results
    }

DIR_ROOT=os.environ['DIR_ROOT']

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
#
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
    args.iscuda = common.torch_set_gpu(args.gpu,seed=43)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

#    dl.download_dataset(args.dataset)
#    dataset = datasets.create('Paris6K_train')
    
    device = torch.device('cuda' if args.iscuda else 'cpu')
    net = load_model(args.checkpoint, args.iscuda)
#    net=InceptionResnetV1(pretrained='vggface2')

    net=iresnet50(pretrained=False, fp16=False)
    net.load_state_dict(torch.load(os.path.join(DIR_ROOT, 'dirtorch/data/models/backbone.pth'),map_location=device))
    net=common.switch_model_to_cuda(net, args.iscuda)
#    optim=Adam(net.parameters(),lr=0.00001)
#    optim = Adam(
#        net.parameters(),
#        lr=1e-5,           # 低学习率
#        betas=(0.5, 0.999), # 调整beta1和beta2
#        weight_decay=1e-5   # L2正则化
#    )
    optim=torch.optim.SGD(
    net.parameters(),
    lr=1e-4,           # SGD需要更大的初始学习率（通常比Adam大10倍）
    momentum=0.9,       # 动量系数（0.9是常用值）
    weight_decay=1e-4   # L2正则化（可适当增大）
)
    scheduler = StepLR(optim, step_size=10, gamma=0.1)  # 每10个epoch学习率衰减为原来的0.1倍
#    if os.path.exists(r'/kaggle/working/resnet_triplet_optim.pt'):
#        optim.load_state_dict(torch.load(r'/kaggle/working/resnet_triplet_optim.pt'))
#    criterion=nn.TripletMarginLoss()
    criterion_nce=TripletLoss(margin=0.3, hard_mining=False)#在线Hard Triplet Loss
#    criterion_nce=InfoNCELoss()
#    criterion_arc=ArcFaceLoss(embedding_dim=512, num_classes=1200)
    criterion_nce = criterion_nce.to(device) # <-- 添加这一行，将损失函数移动到GPU
#    criterion_arc = criterion_arc.to(device) # <-- 添加这一行，将损失函数移动到GPU

    scheduler = StepLR(optim, step_size=10, gamma=0.1)  # 每10个epoch学习率衰减为原来的0.1倍
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
    
    Epoch_num=1
    start_time=time.time()
    Ave_loss=0
    last_five_train_map=np.zeros(5)
    train_map=[]
    last_five_eval_map=np.zeros(5)
    eval_map1=[]
    eval_map2=[]
    eval_map3=[]
    p1,p2,p3=[],[],[]
    train_p=[]
    best_map=0.0

    data=[]
    labels=[]
    dataset=datasets.create('CADA2000_3')
    qdataset=dataset.get_query_db()
    tag=np.zeros(2002)
    data_query=[]
    label_query=[]
    data_query=[trf(qdataset.get_image(i)) for i in range(dataset.nquery)]
    label_query=[dataset.index[i] for i in range(dataset.nquery)]
    for idx in range(dataset.nquery):
        if tag[int(dataset.index2[dataset.relevants[idx][0]])]==1 or len(dataset.relevants[idx])==0:
            continue
        data.extend([trf(dataset.get_image(i)) for i in dataset.relevants[idx]])
        labels.extend([dataset.index2[i] for i in dataset.relevants[idx]])
        tag[int(dataset.index2[dataset.relevants[idx][0]])]=1
    data.extend(data_query)
    labels.extend(label_query)
    data=torch.stack(data).to(device)
    labels=torch.tensor(labels).to(device)
    tsne_stats = visualize_tsne(3,data, labels, net, dataset, 
                                    n_labels=10, 
                                    n_samples_per_label=5, 
                                    save_dir='/kaggle/working')
    print(f"\nt-SNE统计信息:")
    print(f"选中的标签: {tsne_stats['selected_labels']}")
    '''
    # 1. 基础注意力热力图可视化（10个随机样本）
    print("\n生成注意力热力图...")
#    net3=get_mtl_model()
    net3 = mobilefacenet(
        pretrained=os.path.join(DIR_ROOT, r'dirtorch/data/models/Epoch_17ddd.pt'),  # 如果有预训练权重，填写路径
        embedding_dim=512,
        input_size=(112, 112)  # 或 (224, 224)
    )
    net3=common.switch_model_to_cuda(net3, args.iscuda)
    net2 = mobilefacenet(
        pretrained=os.path.join(DIR_ROOT, r'dirtorch/data/models/resnet_triplet.pt'),  # 如果有预训练权重，填写路径
        embedding_dim=512,
        input_size=(112, 112)  # 或 (224, 224)
    )
    net2=common.switch_model_to_cuda(net2, args.iscuda)
#    visualize_attention_heatmap(data, labels, net, model_names='Baseline', sample_idx=0)

    # 多个模型对比
    visualize_attention_heatmap(
        data, labels,
        models=[net, net2, net3],
        model_names=['Baseline', 'TAL', 'IAL'],
        target_layer_names=['layer4', 'block8', 'conv_6_sep'],
        sample_idx=0
    )
    '''
    net4=InceptionResnetV1(pretrained='vggface2')
    net4=common.switch_model_to_cuda(net4, args.iscuda)
    
    # 绘制类内类间距离分布
    dist_stats = plot_intra_inter_class_distance_distribution(
        data, labels, net4, save_dir='/kaggle/working',
        max_samples_per_class=30,  # 每类最多30个样本
        max_pairs=5000,            # 各计算5000对距离
        feature_batch_size=32      # 如果还超显存,改为32或16
    )
    print(f"\n距离分布统计:")
    print(f"类内距离均值: {dist_stats['intra_mean']:.4f}")
    print(f"类间距离均值: {dist_stats['inter_mean']:.4f}")
    print(f"均值差距: {dist_stats['mean_gap']:.4f}")
    

    net.train()
    for epoch in range(Epoch_num):
        dataset = datasets.create('CADA2000_train')#创建数据集对象
        net.train()
        loader = create_pk_dataloader(dataset, p=16, k=8, num_workers=args.threads)#创建P-K采样的数据加载器
        device=torch.device('cuda' if args.iscuda else 'cpu')
        cnt=0
        Ave_loss=num_batches=0
        for batch_idx, (data, labels) in tqdm.tqdm(enumerate(loader), total=1000):
            # 由于我们的数据加载器返回整个batch，需要squeeze
            #print(data.shape)
            if batch_idx>2:
                break
            data = data.squeeze(0).to(device)  # [batch_size, 1, 28, 28]
            labels = labels.squeeze(0).to(device)  # [batch_size]
            triplet_stats = visualize_triplets_with_model(data, labels, net, dataset, margin=0.3, save_dir='/kaggle/working')
            
    
    data=[]
    labels=[]
    dataset=datasets.create('CADA2000_2')
    tag=np.zeros(2002)
    for idx in range(dataset.nquery):
        if tag[int(dataset.index2[dataset.relevants[idx][0]])]==1 or len(dataset.relevants[idx])==0:
            continue
        data.extend([trf(dataset.get_image(i)) for i in dataset.relevants[idx]])
        labels.extend([dataset.index2[i] for i in dataset.relevants[idx]])
        tag[int(dataset.index2[dataset.relevants[idx][0]])]=1
    data.extend(data_query)
    labels.extend(label_query)
    data=torch.stack(data).to(device)
    labels=torch.tensor(labels).to(device)
    tsne_stats = visualize_tsne(2,data, labels, net, dataset, 
                                    n_labels=10, 
                                    n_samples_per_label=10, 
                                    save_dir='/kaggle/working')
    print(f"\nt-SNE统计信息:")
    print(f"选中的标签: {tsne_stats['selected_labels']}")

    data=[]
    labels=[]
    dataset=datasets.create('CADA2000_1')
    tag=np.zeros(2002)
    for idx in range(dataset.nquery):
        if tag[int(dataset.index2[dataset.relevants[idx][0]])]==1 or len(dataset.relevants[idx])==0:
            continue
        data.extend([trf(dataset.get_image(i)) for i in dataset.relevants[idx]])
        labels.extend([dataset.index2[i] for i in dataset.relevants[idx]])
        tag[int(dataset.index2[dataset.relevants[idx][0]])]=1
    data.extend(data_query)
    labels.extend(label_query)
    data=torch.stack(data).to(device)
    labels=torch.tensor(labels).to(device)
    tsne_stats = visualize_tsne(1,data, labels, net, dataset, 
                                    n_labels=10, 
                                    n_samples_per_label=10, 
                                    save_dir='/kaggle/working')
    print(f"\nt-SNE统计信息:")
    print(f"选中的标签: {tsne_stats['selected_labels']}")

    cleanup_and_save(net, optim, output_dir='/kaggle/working', max_run_time=40)