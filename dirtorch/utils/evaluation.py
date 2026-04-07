'''Evaluation metrics
'''
import pdb
import numpy as np
import torch


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k

    output: torch.FloatTensoror np.array(float)
            shape = B * L [* H * W]
            L: number of possible labels

    target: torch.IntTensor or np.array(int)
            shape = B     [* H * W]
            ground-truth labels
    """
    if isinstance(output, np.ndarray):
        pred = (-output).argsort(axis=1)
        target = np.expand_dims(target, axis=1)
        correct = (pred == target)

        res = []
        for k in topk:
            correct_k = correct[:, :k].sum()
            res.append(correct_k / target.size)

    if isinstance(output, torch.Tensor):
        _, pred = output.topk(max(topk), 1, True, True)
        correct = pred.eq(target.unsqueeze(1))

        res = []
        for k in topk:
            correct_k = correct[:, :k].float().view(-1).sum(0)
            res.append(correct_k.mul_(1.0 / target.numel()))

    return res


def compute_AP(label, score):
    from sklearn.metrics import average_precision_score
    return average_precision_score(label, score)


def compute_average_precision(positive_ranks):#AP的计算，结合Precision-Recall曲线
    """
    Extracted from: https://github.com/tensorflow/models/blob/master/research/delf/delf/python/detect_to_retrieve/dataset.py

    Computes average precision according to dataset convention.
    It assumes that `positive_ranks` contains the ranks for all expected positive
    index images to be retrieved. If `positive_ranks` is empty, returns
    `average_precision` = 0.
    Note that average precision computation here does NOT use the finite sum
    method (see
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
    which is common in information retrieval literature. Instead, the method
    implemented here integrates over the precision-recall curve by averaging two
    adjacent precision points, then multiplying by the recall step. This is the
    convention for the Revisited Oxford/Paris datasets.
    Args:
        positive_ranks: Sorted 1D NumPy integer array, zero-indexed.
    Returns:
        average_precision: Float.
    """
    average_precision = 0.0

    num_expected_positives = len(positive_ranks)
    if not num_expected_positives:
        return average_precision

    recall_step = 1.0 / num_expected_positives#召回率计算公式中的分母是定值
    for i, rank in enumerate(positive_ranks):
        if not rank:
            left_precision = 1.0
        else:
            left_precision = i / rank#精确率

        right_precision = (i + 1) / (rank + 1)#精确率
        average_precision += (left_precision + right_precision) * recall_step / 2#采用相邻点的平均

    p1 = 1.0 if positive_ranks[0]==0 else 0.0
    return average_precision,p1#改过了，原来只有average_precision

def compute_precision_at_k(positive_ranks, max_k=None):#新加的，可计算rank-k的precision
    """
    计算Precision@k值，返回前k个位置的precision@k数组
    
    Args:
        positive_ranks: Sorted 1D NumPy integer array, zero-indexed.
                       记录了正确查询结果的排名（从小到大排序）
        max_k: 最大的k值，如果为None则使用positive_ranks中的最大值
    
    Returns:
        precision_at_k: 1D NumPy array，包含P@1, P@2, ..., P@max_k的值
    """
    if not len(positive_ranks):
        if max_k is None:
            return np.array([])
        else:
            return np.zeros(max_k)
    
    # 如果没有指定max_k，使用positive_ranks中的最大值+1作为max_k
    if max_k is None:
        max_k = positive_ranks[-1] + 1
    
    precision_at_k = np.zeros(max_k)
    
    # 对于每个k值，计算precision@k
    for k in range(1, max_k + 1):
        # 统计在前k个结果中有多少个是正确的
        num_relevant_in_topk = np.sum(positive_ranks < k)
        precision_at_k[k-1] = num_relevant_in_topk / k
    
    return precision_at_k


def compute_average_precision_quantized(labels, idx, step=0.01):
    recall_checkpoints = np.arange(0, 1, step)

    def mymax(x, default):
        return np.max(x) if len(x) else default

    Nrel = np.sum(labels)
    if Nrel == 0:
        return 0
    recall = np.cumsum(labels[idx])/float(Nrel)
    irange = np.arange(1, len(idx)+1)
    prec = np.cumsum(labels[idx]).astype(np.float32) / irange
    precs = np.array([mymax(prec[np.where(recall > v)], 0) for v in recall_checkpoints])
    return np.mean(precs)


def pixelwise_iou(output, target):
    """ For each image, for each label, compute the IoU between
    """
    assert False
