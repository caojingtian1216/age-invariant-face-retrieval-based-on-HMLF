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
import math

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
from dirtorch.nets.Swin_Transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
from dirtorch.nets.MobileFaceNets import mobilefacenet
from dirtorch.config import cfg

trf=transforms.Compose([
    transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),#将图像调整为224x224大小
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
])
def plot_map(train_map_list,eval_map_list):
    plt.figure(figsize=(10, 5))
    plt.plot(train_map_list, label='Train mAP', marker='o')
    plt.plot(eval_map_list, label='Eval mAP', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.legend()
    plt.grid(True)
    
    #保存
    plt.savefig('/kaggle/working/map_plot.jpg')
    plt.show()
def plot_p(train_p_list,eval_p_list):
    plt.figure(figsize=(10, 5))
    plt.plot(train_p_list, label='Train P@1', marker='o')
    plt.plot(eval_p_list, label='Eval P@1', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('P@1')
    plt.title('P@1 over Epochs')
    plt.legend()
    plt.grid(True)

    #保存
    plt.savefig('/kaggle/working/p_plot.jpg')
    plt.show()
def visualize_retrieval_results(qdb_idx,query_image, retrieval_images, retrieval_scores=None, retrieval_labels=None, query_label=None):
    titles=['(测试集)','（训练集）']
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

def plot_multiple_cmc_curves(cmc_dict, save_path=None, title='CMC Curves Comparison'):
    """
    绘制多条CMC曲线进行比较
    
    Args:
        cmc_dict: dict, 格式为 {'模型名': [rank_1, rank_2, ...], ...}
        save_path: str, 保存路径
        title: str, 图表标题
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (model_name, accuracies) in enumerate(cmc_dict.items()):
        ranks = np.arange(1, len(accuracies) + 1)
        plt.plot(ranks, accuracies, 
                color=colors[idx % len(colors)], 
                marker=markers[idx % len(markers)],
                linewidth=2, 
                markersize=8, 
                label=f'{model_name} (R1={accuracies[0]:.1f}%)')
    
    plt.xlabel('Rank', fontsize=14, fontweight='bold')
    plt.ylabel('Recognition Rate (%)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='lower right')
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CMC曲线对比图已保存到: {save_path}")
    
    plt.show()

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
        rank_k = [db.eval_rank_ks(q, s) for q, s in enumerate(tqdm.tqdm(scores, desc='rank-k'))]
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
            res['rank_k']=rank_k#新增的，保存rank-k的结果
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
            return self._all_triplet_loss(dist_matrix, labels_equal, labels_not_equal, eye_mask)
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

class ArcFaceLoss(nn.Module):
    """
    ArcFace 损失函数的简单实现。
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        """
        Args:
            embedding_dim (int): 特征向量的维度。
            num_classes (int): 训练集中的总类别数。
            s (float): 半径缩放因子。论文中推荐值为 64，这里用 30 作为示例。
            m (float): 角度边距 (additive angular margin)。
        """
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # 创建分类器权重，它将作为每个类别的中心
        # nn.Parameter 会将这个张量注册为模型的可训练参数
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        # 对权重进行良好的初始化
        nn.init.xavier_uniform_(self.weight)

        # 用于计算角度边距的辅助常量
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # 阈值，用于防止角度超过 pi
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        前向传播。

        Args:
            embeddings (torch.Tensor): 来自模型的特征向量，形状为 [batch_size, embedding_dim]。
            labels (torch.Tensor): 对应的真实标签，形状为 [batch_size]。

        Returns:
            torch.Tensor: 计算出的损失值。
        """
        # 1. L2 归一化特征向量和分类器权重
        # 使得它们的点积等于余弦相似度
        embeddings_normalized = F.normalize(embeddings)
        weights_normalized = F.normalize(self.weight)

        # 2. 计算特征向量与所有类别中心的余弦相似度
        # cosine 的形状为 [batch_size, num_classes]
        cosine = F.linear(embeddings_normalized, weights_normalized)

        # 3. 计算角度 theta
        # 为了数值稳定性，使用 clamp 将值限制在 [-1, 1] 之间
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # 4. 计算添加了角度边距 m 后的目标 logit: cos(theta + m)
        # 使用三角函数公式: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # ArcFace 的一个技巧：如果 theta + m > 180度，则使用一个简化的惩罚项
        # 这可以防止梯度在 theta 很小的时候变得过大，有助于稳定训练
        # 当 cosine > self.th 时，表示 theta + m < 180度
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. 将添加了边距的 logit 更新到原始的 cosine 矩阵中
        # one_hot 编码用于定位每个样本对应的正确类别
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        # scatter_ 会在 one_hot 矩阵的指定位置（由 labels 决定）填充 1
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # output 中，正确类别的 logit 是 phi，其他类别的 logit 仍然是 cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. 对所有 logit 进行缩放
        output *= self.s

        # 7. 使用标准的交叉熵损失函数计算最终损失
        loss = F.cross_entropy(output, labels)

        return loss
def print_res(res):
    count=0
    for key, value in res.items():
        print(f'{key}: {value}')
        count += 1
        if count == 2:
            break
class ModelWithUncertaintyWeight(nn.Module):
    def __init__(self, your_main_model):
        super().__init__()
        self.model = your_main_model # 你的主模型
        # 初始化两个可学习的不确定性参数。通常初始化为0，使得初始权重为1/(2*exp(0)^2)=0.5
        self.log_var1 = nn.Parameter(torch.zeros(1))
        self.log_var2 = nn.Parameter(torch.zeros(1))
        self.rgb_means = cfg.MEAN
        self.rgb_stds  = cfg.STD
        self.input_size = (3,cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        self.preprocess = dict(
            mean=self.rgb_means,
            std=self.rgb_stds,
            input_size=cfg.INPUT_SIZE
        )
        self.iscuda = torch.cuda.is_available()

    def forward(self, x):
        return self.model(x)

    def get_total_loss(self, loss1, loss2):
        # 使用 log(var) 而不是直接使用 var，是为了训练更稳定（避免除以零和数值问题）
        # 权重 = 1 / (2 * exp(log_var)^2) = 1 / (2 * var)
        # 但我们在实现中直接用 log_var 来计算，等价于上式。
        weight1 = 1.0 / (2.0 * torch.exp(self.log_var1))
        weight2 = 1.0 / (2.0 * torch.exp(self.log_var2))

        # 总损失 = 加权损失 + 正则项
        total_loss = weight1 * loss1 + weight2 * loss2 + self.log_var1 + self.log_var2
        # 你也可以给正则项加一个小的系数，如 0.5 * (self.log_var1 + self.log_var2)
        return total_loss
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
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
#    parser.add_argument('--gpu', type=int, default=-1, nargs='+', help='GPU ids')
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

    device = torch.device('cuda' if args.iscuda else 'cpu')
    net = load_model(args.checkpoint, args.iscuda)
#    net=InceptionResnetV1(pretrained='vggface2')

#    net=iresnet50(pretrained=False, fp16=False)
#    net.load_state_dict(torch.load(os.path.join(DIR_ROOT, 'dirtorch/data/models/backbone.pth'),map_location=device))
#    net=get_model(model_name='convnext_tiny', pretrained=False, embedding_dim=512)
#    net=get_mtl_model()
#    net=ModelWithUncertaintyWeight(net)
    net=swin_tiny_patch4_window7_224(pretrained=os.path.join(DIR_ROOT, r'dirtorch/data/models/swin_tiny_patch4_window7_224.pt'))
#    net = mobilefacenet(
#        pretrained=os.path.join(DIR_ROOT, r'dirtorch/data/models/Epoch_17ddd.pt'),  # 如果有预训练权重，填写路径
#        embedding_dim=512,
#        input_size=(112, 112)  # 或 (224, 224)
#    )
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
#    criterion_nce=InfoNCELoss()#65536
    criterion_arc=ArcFaceLoss(embedding_dim=512, num_classes=507)
    criterion_nce = criterion_nce.to(device) # <-- 添加这一行，将损失函数移动到GPU
    criterion_arc = criterion_arc.to(device) # <-- 添加这一行，将损失函数移动到GPU

    optim = torch.optim.SGD([
    # Backbone 用小学习率 (比如 1e-4)
    {'params': net.parameters(), 'lr': 1e-4},
    # Head 用大学习率 (比如 1e-3)
    {'params': criterion_arc.parameters(), 'lr': 1e-3}
], momentum=0.9, weight_decay=1e-4)
#    optim=torch.optim.AdamW(
#        net.parameters(),
#        lr=5e-3,           # 低学习率
#        weight_decay=1e-4,   # L2正则化
#        betas=(0.9, 0.999)
#    )
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
    
    Epoch_num=60
    start_time=time.time()
    Ave_loss=0
    last_five_train_map=np.zeros(5)
    train_map=[]
    last_five_eval_map=np.zeros(5)
    eval_map=[]
    eval_p=[]
    train_p=[]
    best_map=0.0
    
    dataset = datasets.create('agedb_trainvalid')
    qdataset=dataset.get_query_db()
    net.eval()
    args.detailed=False
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                    threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                    save_feats=args.save_feats, load_feats=args.load_feats)
    print_res(res)
    for idx in range(3):
        retrieval_images=[dataset.get_image(i) for i in res['results'][idx]]
        labels=[dataset.index2[i] for i in res['results'][idx]]
        qlabel=dataset.index[idx]
        visualize_retrieval_results(1,qdataset.get_image(idx),retrieval_images,res['scores'][idx],labels,qlabel)
    train_map.append(res['mAP'])
    train_p.append(res['P@1'])

    dataset = datasets.create('agedb')
    qdataset=dataset.get_query_db()
    net.eval()
    args.detailed=False
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                    threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                    save_feats=args.save_feats, load_feats=args.load_feats)
    print_res(res)
    rank_k_np=np.array(res['rank_k'])
    ave_rank_k=rank_k_np.mean(axis=0)
    print("各个Rank-K的平均值：",ave_rank_k)

    for idx in range(3):
        retrieval_images=[dataset.get_image(i) for i in res['results'][idx]]
        labels=[dataset.index2[i] for i in res['results'][idx]]
        qlabel=dataset.index[idx]
        visualize_retrieval_results(0,qdataset.get_image(idx),retrieval_images,res['scores'][idx],labels,qlabel)
    eval_map.append(res['mAP'])
    eval_p.append(res['P@1'])

    net.train()
    for epoch in range(Epoch_num):
        if epoch==0:
            for i in net.parameters():#kaggle上需要改一下net.module.fc.parameters()
                i.requires_grad = True
            
        #loader = get_loader(dataset, trf_chain=args.trfs, preprocess=net.preprocess, iscuda=args.iscuda,
        #                    output=['img'], batch_size=args.batch_size, threads=args.threads, training=True, shuffle=True)#只要图片，得到DataLoader对象
        dataset = datasets.create('agedb_train')#创建数据集对象
        net.train()
        p,k=16,4
        loader = create_pk_dataloader(dataset, p=p, k=k, num_workers=args.threads)#创建P-K采样的数据加载器
        device=torch.device('cuda' if args.iscuda else 'cpu')
        cnt=0
        Ave_loss=num_batches=0
        for batch_idx, (data, labels) in tqdm.tqdm(enumerate(loader), total=2304/(p*k)):#2304是morph_train的图片数量
            # 由于我们的数据加载器返回整个batch，需要squeeze
            #print(data.shape)
            data = data.squeeze(0).to(device)  # [batch_size, 1, 28, 28]
            labels = labels.squeeze(0).to(device)  # [batch_size]
            
            optim.zero_grad()
            
            # 前向传播
            embeddings = net(data)
            loss_arc = criterion_arc(embeddings, labels)
            loss_nce = criterion_nce(embeddings, labels)
            #loss = 0.2*loss_arc + loss_nce
            loss=loss_nce
            #loss=net.get_total_loss(loss_nce, loss_arc)
                # 权重 w = 1/(2*exp(log_var))， 不确定性 sigma = exp(log_var)
            #learned_weight1 = 1.0 / (2.0 * torch.exp(net.log_var1).item())
            #learned_weight2 = 1.0 / (2.0 * torch.exp(net.log_var2).item())
            
            
            # 反向传播
            loss_nce.backward()

            for param_group in optim.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=1.0)
            optim.step()
            
            Ave_loss += loss.item()
            num_batches += 1

            if time.time()-start_time>3600*3:
                elapsed = time.time() - start_time
                plot_map(train_map, eval_map)
                plot_p(train_p, eval_p)
                print(f"即将超时（已运行 {elapsed/3600:.2f} 小时），开始清理...")
                cleanup_and_save(net, optim)

            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{Epoch_num}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
            #    print(f"Task1 Weight: {learned_weight1:.4f}, Task2 Weight: {learned_weight2:.4f}")
            
            # 限制每个epoch的batch数量，避免训练时间过长
            if num_batches >= 1000:
                break
        print("Epoch %d Average Loss: %.4f"%(epoch,Ave_loss/num_batches))
        scheduler.step()


        #训练部分的代码***
        # Evaluate
    #    dataset = datasets.create('Paris6K')
        
        dataset = datasets.create('agedb_trainvalid')
        qdataset=dataset.get_query_db()
        net.eval()
        args.detailed=False
        res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                        threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                        save_feats=args.save_feats, load_feats=args.load_feats)
        print_res(res)
        for idx in range(3):
            retrieval_images=[dataset.get_image(i) for i in res['results'][idx]]
            labels=[dataset.index2[i] for i in res['results'][idx]]
            qlabel=dataset.index[idx]
            visualize_retrieval_results(0,qdataset.get_image(idx),retrieval_images,res['scores'][idx],labels,qlabel)
        last_five_train_map[epoch%5]=res['mAP']
        train_map.append(res['mAP'])
        train_p.append(res['P@1'])
        
        dataset = datasets.create('agedb')
        qdataset=dataset.get_query_db()
        net.eval()
        args.detailed=False
        res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                        threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                        save_feats=args.save_feats, load_feats=args.load_feats)
    #    print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))
        print_res(res)
        rank_k_np=np.array(res['rank_k'])
        ave_rank_k=rank_k_np.mean(axis=0)
        print("各个Rank-K的平均值：",ave_rank_k)
        plot_multiple_cmc_curves({'AgeDB': ave_rank_k*100},save_path=r'/kaggle/working/cmc_curve.jpg')
        
        for idx in range(3):
            retrieval_images=[dataset.get_image(i) for i in res['results'][idx]]
            labels=[dataset.index2[i] for i in res['results'][idx]]
            qlabel=dataset.index[idx]
            visualize_retrieval_results(1,qdataset.get_image(idx),retrieval_images,res['scores'][idx],labels,qlabel)
        last_five_eval_map[epoch%5]=res['mAP']
        eval_map.append(res['mAP'])
        eval_p.append(res['P@1'])
        best_map=max(last_five_eval_map.max(), best_map)
        #if last_five_eval_map.max()< best_map or abs(last_five_eval_map[epoch%5]-last_five_train_map[epoch%5])>0.4:
        if last_five_eval_map.max()< best_map:
            print("Best mAP so far: %.4f"%(best_map))
            print(f"Early stopped at epoch {epoch}")
            plot_map(train_map, eval_map)
            plot_p(train_p, eval_p)
            cleanup_and_save(net, optim)

    plot_map(train_map, eval_map)
    plot_p(train_p, eval_p)
    cleanup_and_save(net, optim)

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
    