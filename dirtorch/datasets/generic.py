import os
import json
import pdb
import numpy as np
import pickle
import os.path as osp
import json

from .dataset import Dataset
from .generic_func import *


class ImageList(Dataset):
    ''' Just a list of images (no labels, no query).

    Input:  text file, 1 image path per row
    '''
    def __init__(self, img_list_path, root='', imgs=None):
        self.root = root
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = [e.strip() for e in open(img_list_path)]

        self.nimg = len(self.imgs)
        self.nclass = 0
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]


class LabelledDataset (Dataset):
    """ A dataset with per-image labels
        and some convenient functions.
    """
    def find_classes(self, *arg, **cls_idx):
        labels = arg[0] if arg else self.labels
        self.classes, self.cls_idx = find_and_list_classes(labels, cls_idx=cls_idx)
        self.nclass = len(self.classes)
        self.c_relevant_idx = find_relevants(self.labels)


class ImageListLabels(LabelledDataset):
    ''' Just a list of images with labels (no queries).

    Input:  text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, img_list_path, root=None):
        self.root = root
        if osp.splitext(img_list_path)[1] == '.txt':
            tmp = [e.strip() for e in open(img_list_path)]
            self.imgs = [e.split(' ')[0] for e in tmp]
            self.labels = [e.split(' ')[1] for e in tmp]
        elif osp.splitext(img_list_path)[1] == '.json':
            d = json.load(open(img_list_path))
            self.imgs = []
            self.labels = []
            for i, l in d.items():
                self.imgs.append(i)
                self.labels.append(l)
        self.find_classes()

        self.nimg = len(self.imgs)
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_label(self, i, toint=False):
        label = self.labels[i]
        if toint:
            label = self.cls_idx[label]
        return label

    def get_query_db(self):
        return self


class ImageListLabelsQ(ImageListLabels):
    ''' Two list of images with labels: one for the dataset and one for the queries.

    Input:  text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, img_list_path, query_list_path, root=None):
        self.root = root
        tmp = [e.strip() for e in open(img_list_path)]
        self.imgs = [e.split(' ')[0] for e in tmp]
        self.labels = [e.split(' ')[1] for e in tmp]
        tmp = [e.strip() for e in open(query_list_path)]
        self.qimgs = [e.split(' ')[0] for e in tmp]
        self.qlabels = [e.split(' ')[1] for e in tmp]
        self.find_classes()

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)

    def find_classes(self, *arg, **cls_idx):
        labels = arg[0] if arg else self.labels + self.qlabels
        self.classes, self.cls_idx = find_and_list_classes(labels, cls_idx=cls_idx)
        self.nclass = len(self.classes)
        self.c_relevant_idx = find_relevants(self.labels)

    def get_query_db(self):
        return ImagesAndLabels(self.qimgs, self.qlabels, self.cls_idx, root=self.root)


class ImagesAndLabels(ImageListLabels):
    ''' Just a list of images with labels.

    Input:  two arrays containing the text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, imgs, labels, cls_idx, root=None):
        self.root = root
        self.imgs = imgs
        self.labels = labels
        self.cls_idx = cls_idx
        self.nclass = len(self.cls_idx.keys())

        self.nimg = len(self.imgs)
        self.nquery = 0


class ImageListRelevants(Dataset):
    """ A dataset composed by a list of images, a list of indices used as queries,
        and for each query a list of relevant and junk indices (ie. Oxford-like GT format)

        Input: path to the pickle file
    """
    def __init__(self, gt_file, root=None, img_dir='jpg', ext='.jpg'):
        self.root = root
        self.img_dir = img_dir

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)
        self.imgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['imlist']]
        self.qimgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['qimlist']]
        self.qroi = [tuple(e['bbx']) for e in gt['gnd']]
        if 'ok' in gt['gnd'][0]:
            self.relevants = [e['ok'] for e in gt['gnd']]
            self.irrelevants=[]
            irrelevants_lst=[]
            for i in range(len(self.qimgs)):
                irrelevants_lst=[]
                for j in range(0,len(self.qimgs),5):
                    if i//5!=j//5:
                        irrelevants_lst.extend(self.relevants[j])
                self.irrelevants.append(irrelevants_lst)
        else:
            self.relevants = None
            self.easy = [e['easy'] for e in gt['gnd']]
            self.hard = [e['hard'] for e in gt['gnd']]
        self.junk = [e['junk'] for e in gt['gnd']]

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)


    def get_relevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            rel = self.relevants[qimg_idx]
        elif mode == 'easy':
            rel = self.easy[qimg_idx]
        elif mode == 'medium':
            rel = self.easy[qimg_idx] + self.hard[qimg_idx]#调整难度
        elif mode == 'hard':
            rel = self.hard[qimg_idx]
        return rel
    
    #返回所有不相关的图片
    def get_irrelevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            irrel=self.irrelevants[qimg_idx]
        return irrel

    def get_junk(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            junk = self.junk[qimg_idx]
        elif mode == 'easy':
            junk = self.junk[qimg_idx] + self.hard[qimg_idx]#为什么要混入hard的
        elif mode == 'medium':
            junk = self.junk[qimg_idx]
        elif mode == 'hard':
            junk = self.junk[qimg_idx] + self.easy[qimg_idx]
        return junk

    def get_query_filename(self, qimg_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_query_key(qimg_idx))

    def get_query_roi(self, qimg_idx):
        return self.qroi[qimg_idx]

    def get_key(self, i):#返回文件名
        return self.imgs[i]
    
    ##补的找类别
    def get_img_class(self,i):
        return self.imgs[i].rsplit('_',1)[0]

    def get_query_key(self, i):
        return self.qimgs[i]

    def get_query_db(self):
        return ImageListROIs(self.root, self.img_dir, self.qimgs, self.qroi)

    def get_query_groundtruth(self, query_idx, what='AP', mode='classic'):
        # negatives
        res = -np.ones(self.nimg, dtype=np.int8)
        # positive
        res[self.get_relevants(query_idx, mode)] = 1
        # junk
        res[self.get_junk(query_idx, mode)] = 0
        return res

    def eval_query_AP(self, query_idx, scores):#
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels去掉不相关的

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]#第一个label是1的下标“们”，[0]是np.where的固定用法，因为返回的是元组eg.([1,2],)
            return compute_average_precision(positive_rank)
        else:
            d = {}
            for mode in ('easy', 'medium', 'hard'):
                gt = self.get_query_groundtruth(query_idx, 'AP', mode)  # labels in {-1, 0, 1}
                assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
                assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
                keep = (gt != 0)  # remove null labels
                if sum(gt[keep] > 0) == 0:  # exclude queries with no relevants from the evaluation
                    d[mode] = -1
                else:
                    gt2, scores2 = gt[keep], scores[keep]
                    gt_sorted = gt2[np.argsort(scores2)[::-1]]
                    positive_rank = np.where(gt_sorted == 1)[0]
                    d[mode] = compute_average_precision(positive_rank)
            return d
        
class ImageListRelevants2(Dataset):
    """ A dataset composed by a list of images, a list of indices used as queries,
        and for each query a list of relevant and junk indices (ie. Oxford-like GT format)

        Input: path to the pickle file
    """
    def __init__(self, gt_file, root=None, img_dir='jpg', ext='.jpg'):
        self.root = root
        self.img_dir = img_dir

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)
        self.imgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['imlist']]
        self.qimgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['qimlist']]
        self.qroi = [tuple(e['bbx']) for e in gt['gnd']]
        self.index2=np.zeros(len(self.imgs))
        if 'ok' in gt['gnd'][0]:
            self.relevants = [e['ok'] for e in gt['gnd']]
            self.irrelevants=[e['irrel'] for e in gt['gnd']]
            self.index=gt['index']
            for i in range(len(self.qimgs)):
                for j in self.relevants[i]:
                    self.index2[j]=self.index[i]
            if 'imlabel' in gt:
                self.imlabel = gt['imlabel']
                self.index2=np.array(self.imlabel)
        else:
            self.relevants = None
            self.easy = [e['easy'] for e in gt['gnd']]
            self.hard = [e['hard'] for e in gt['gnd']]
        self.junk = [e['junk'] for e in gt['gnd']]

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)


    def get_relevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            rel = self.relevants[qimg_idx]
        elif mode == 'easy':
            rel = self.easy[qimg_idx]
        elif mode == 'medium':
            rel = self.easy[qimg_idx] + self.hard[qimg_idx]#调整难度
        elif mode == 'hard':
            rel = self.hard[qimg_idx]
        return rel
    
    #返回所有不相关的图片
    def get_irrelevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            irrel=self.irrelevants[self.index[qimg_idx]-1]#注意下标减一
        return irrel

    def get_junk(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            junk = self.junk[qimg_idx]
        elif mode == 'easy':
            junk = self.junk[qimg_idx] + self.hard[qimg_idx]#为什么要混入hard的
        elif mode == 'medium':
            junk = self.junk[qimg_idx]
        elif mode == 'hard':
            junk = self.junk[qimg_idx] + self.easy[qimg_idx]
        return junk

    def get_query_filename(self, qimg_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_query_key(qimg_idx))

    def get_query_roi(self, qimg_idx):
        return self.qroi[qimg_idx]

    def get_key(self, i):#返回文件名
        return self.imgs[i]
    
    ##补的找类别
    def get_img_class(self,i):
        return self.imgs[i].rsplit('_',1)[0]

    def get_query_key(self, i):
        return self.qimgs[i]

    def get_query_db(self):
        return ImageListROIs(self.root, self.img_dir, self.qimgs, self.qroi)

    def get_query_groundtruth(self, query_idx, what='AP', mode='classic'):
        # negatives
        res = -np.ones(self.nimg, dtype=np.int8)
        # positive
        res[self.get_relevants(query_idx, mode)] = 1
        # junk
        res[self.get_junk(query_idx, mode)] = 0
        return res

    def eval_query_AP(self, query_idx, scores):#
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels去掉不相关的

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]#第一个label是1的下标“们”，[0]是np.where的固定用法，因为返回的是元组eg.([1,2],)
            return compute_average_precision(positive_rank),np.argsort(scores)[::-1],np.sort(scores)[::-1] #改过了，为了画图
        else:
            d = {}
            for mode in ('easy', 'medium', 'hard'):
                gt = self.get_query_groundtruth(query_idx, 'AP', mode)  # labels in {-1, 0, 1}
                assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
                assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
                keep = (gt != 0)  # remove null labels
                if sum(gt[keep] > 0) == 0:  # exclude queries with no relevants from the evaluation
                    d[mode] = -1
                else:
                    gt2, scores2 = gt[keep], scores[keep]
                    gt_sorted = gt2[np.argsort(scores2)[::-1]]
                    positive_rank = np.where(gt_sorted == 1)[0]
                    d[mode] = compute_average_precision(positive_rank)
            return d
    def eval_rank_ks(self, query_idx, scores):#
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision,compute_precision_at_k
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels去掉不相关的

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]#第一个label是1的下标“们”，[0]是np.where的固定用法，因为返回的是元组eg.([1,2],)
            return compute_precision_at_k(positive_rank,max_k=10)

class ImageListRelevants3(Dataset):
    """ A dataset composed by a list of images, a list of indices used as queries,
        and for each query a list of relevant and junk indices (ie. Oxford-like GT format)

        Input: path to the pickle file
    """
    def __init__(self, gt_file, root=None, img_dir='jpg', ext='.jpg'):
        self.root = root
        self.img_dir = img_dir

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)
        self.imgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['imlist']]
        self.qimgs = [e for e in gt['qimlist']]
        self.qroi = [tuple(e['bbx']) for e in gt['gnd']]
        if 'ok' in gt['gnd'][0]:
            cnt=0
            self.relevants = [list(e['ok']) for e in gt['gnd']]
            #print(type(self.relevants[0]))
            self.label=[0 for _ in range(len(self.imgs))]
            for i in range(len(self.qimgs)):
                if len(self.relevants[i])>0:
                    for j in self.relevants[i]:
                        self.label[j]=cnt
                    cnt+=1#这边把类别连续化了
            self.irrelevants=[e['irrel'] for e in gt['gnd']]
#            self.index=gt['index']
        else:
            self.relevants = None
            self.easy = [e['easy'] for e in gt['gnd']]
            self.hard = [e['hard'] for e in gt['gnd']]
        self.junk = [e['junk'] for e in gt['gnd']]

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)


    def get_relevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            rel = self.relevants[qimg_idx]
        elif mode == 'easy':
            rel = self.easy[qimg_idx]
        elif mode == 'medium':
            rel = self.easy[qimg_idx] + self.hard[qimg_idx]#调整难度
        elif mode == 'hard':
            rel = self.hard[qimg_idx]
        return rel
    
    #返回所有不相关的图片
    def get_irrelevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            irrel=self.irrelevants[qimg_idx]
        return irrel

    def get_junk(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            junk = self.junk[qimg_idx]
        elif mode == 'easy':
            junk = self.junk[qimg_idx] + self.hard[qimg_idx]#为什么要混入hard的
        elif mode == 'medium':
            junk = self.junk[qimg_idx]
        elif mode == 'hard':
            junk = self.junk[qimg_idx] + self.easy[qimg_idx]
        return junk

    def get_query_filename(self, qimg_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_query_key(qimg_idx))

    def get_query_roi(self, qimg_idx):
        return self.qroi[qimg_idx]

    def get_key(self, i):#返回文件名
        return self.imgs[i]
    
    ##补的找类别(改过了)
    def get_img_class(self,i):
#        return self.imgs[i].rsplit('_',1)[0]
        return self.label[i]#返回类别的下标

    def get_query_key(self, i):
        return self.qimgs[i]

    def get_query_db(self):
        return ImageListROIs(self.root, self.img_dir, self.qimgs, self.qroi)

    def get_query_groundtruth(self, query_idx, what='AP', mode='classic'):
        # negatives
        res = -np.ones(self.nimg, dtype=np.int8)
        # positive
        res[self.get_relevants(query_idx, mode)] = 1
        # junk
        res[self.get_junk(query_idx, mode)] = 0
        return res

    def eval_query_AP(self, query_idx, scores):#
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels去掉不相关的

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]#第一个label是1的下标“们”，[0]是np.where的固定用法，因为返回的是元组eg.([1,2],)
            return compute_average_precision(positive_rank)
        else:
            d = {}
            for mode in ('easy', 'medium', 'hard'):
                gt = self.get_query_groundtruth(query_idx, 'AP', mode)  # labels in {-1, 0, 1}
                assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
                assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
                keep = (gt != 0)  # remove null labels
                if sum(gt[keep] > 0) == 0:  # exclude queries with no relevants from the evaluation
                    d[mode] = -1
                else:
                    gt2, scores2 = gt[keep], scores[keep]
                    gt_sorted = gt2[np.argsort(scores2)[::-1]]
                    positive_rank = np.where(gt_sorted == 1)[0]
                    d[mode] = compute_average_precision(positive_rank)
            return d

class ImageListROIs(Dataset):#得到qim的信息
    def __init__(self, root, img_dir, imgs, rois):
        self.root = root
        self.img_dir = img_dir
        self.imgs = imgs
        self.rois = rois

        self.nimg = len(self.imgs)
        self.nclass = 0
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_roi(self, i):
        return self.rois[i]

    def get_image(self, img_idx, resize=None):#重载方法，提取边框的ROI区域
        from PIL import Image
        img = Image.open(self.get_filename(img_idx)).convert('RGB')
#        img = img.crop(self.rois[img_idx])#重载的部分，裁剪图像
        if resize:
            img = img.resize(resize, Image.ANTIALIAS if np.prod(resize) < np.prod(img.size) else Image.BICUBIC)
        return img


def not_none(label):
    return label is not None


class ImageClusters(LabelledDataset):
    ''' Just a list of images with labels (no query).

    Input:  JSON, dict of {img_path:class, ...}
    '''
    def __init__(self, json_path, root=None, filter=not_none):
        self.root = root
        self.imgs = []
        self.labels = []
        if isinstance(json_path, dict):
            data = json_path
        else:
            data = json.load(open(json_path))
            assert isinstance(data, dict), 'json content is not a dictionary'

        for img, cls in data.items():
            assert type(img) is str
            if not filter(cls):
                continue
            if type(cls) not in (str, int, type(None)):
                continue
            self.imgs.append(img)
            self.labels.append(cls)

        self.find_classes()
        self.nimg = len(self.imgs)
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_label(self, i, toint=False):
        label = self.labels[i]
        if toint:
            label = self.cls_idx[label]
        return label


class NullCluster(ImageClusters):
    ''' Select only images with null label
    '''
    def __init__(self, json_path, root=None):
        ImageClusters.__init__(self, json_path, root, lambda c: c is None)
