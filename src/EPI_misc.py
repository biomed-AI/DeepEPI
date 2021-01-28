#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
import warnings
import torch
import matplotlib.pyplot as plt

import biock.biock as biock

np.random.seed(0)
torch.manual_seed(0)

def max_indexes(ar):
    ans = list()
    max_val = max(ar)
    for i, x in enumerate(ar):
        if x == max_val:
            ans.append(i)
    return ans
def min_indexes(ar):
    ans = list()
    min_val = min(ar)
    for i, x in enumerate(ar):
        if x == min_val:
            ans.append(i)
    return ans

def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        res = tensor.squeeze().cpu().detach().numpy()
    elif isinstance(tensor, np.ndarray):
        res = tensor
    elif isinstance(tensor, list):
        res = np.array(tensor)
    else:
        warnings.warn("Unexpected type: {}, returned directly".format(type(tensor)))
        res = tensor
    return  res

def cal_auc_torch(y_true, y_prob):
    if type(y_true) is torch.Tensor:
        y_true = tensor2numpy(y_true)
    if type(y_prob) is torch.Tensor:
        y_prob = tensor2numpy(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_prob)
    auroc = auc(fpr, tpr)
    return (auroc, tpr, fpr)

def cal_aupr_torch(y_true, y_prob):
    y_true = tensor2numpy(y_true)
    y_prob = tensor2numpy(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)
    return (aupr, precision, recall)


def cal_f1_torch(y_true, y_prob, cutoff=0.5):
    if type(y_true) is torch.Tensor:
        y_true = tensor2numpy(y_true)
    if type(y_prob) is torch.Tensor:
        y_prod = tensor2numpy(y_prod)
    y_pred = (y_prob >= cutoff).astype(np.int)
    if sum(y_pred) == 0 or sum(y_pred) == len(y_pred):
        f1 = 0
    else:
        f1 = f1_score(y_true, y_pred)
    return f1

def cal_average_precision(y_true, y_prob):
    if type(y_true) is torch.Tensor:
        y_true = tensor2numpy(y_true)
    if type(y_prob) is torch.Tensor:
        y_prob = tensor2numpy(y_prob)
    score = average_precision_score(y_true, y_prob)
    return score

def cal_balanced_accuracy(y_true, y_prob, cutoff=0.5):
    if type(y_true) is torch.Tensor:
        y_true = tensor2numpy(y_true)
    if type(y_prob) is torch.Tensor:
        y_prob = tensor2numpy(y_prob)
    y_pred = (y_prob >= cutoff).astype(np.int)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return balanced_acc

def max_f1(y_true, y_prob):
    max_f1 = -1
    best_cutoff = None
    for cutoff in np.linspace(min(y_prob) + 1E-6, max(y_prob) - 1E-6, 100):
        pred = (y_prob >= cutoff).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > max_f1:
            best_cutoff = cutoff
            max_f1 = f1
    return max_f1, best_cutoff


def evaluate_results(true, prob):
    if type(true) is torch.Tensor:
        true = tensor2numpy(true)
    if type(prob) is torch.Tensor:
        prob = tensor2numpy(prob)
    test_auroc, test_tpr, test_fpr = cal_auc_torch(true.astype(int), prob)
    #test_f1 = cal_f1_torch(true.astype(int), prob)
    test_aupr, precision, recall = cal_aupr_torch(true.astype(int), prob)
    test_ap = cal_average_precision(true.astype(int), prob)
    test_balanced_acc = cal_balanced_accuracy(true.astype(int), prob)
    counts = np.unique(true, return_counts=True)
    true_prob = np.array(list(zip(true, prob)))
    true_prob = np.array(sorted(true_prob, key=lambda l:l[1], reverse=True))
    F1, cutoff = max_f1(true.astype(int), prob)

    num_pos = np.sum(true_prob[:,0])
    if num_pos > 0:
        top10 = np.sum(true_prob[0: int(min(num_pos, 10)), 0]) / min(10, num_pos)
        top100 = np.sum(true_prob[0: int(min(num_pos, 100)), 0]) / min(100, num_pos)
        top500 = np.sum(true_prob[0: int(min(num_pos, 500)), 0]) / min(500, num_pos)
        top1000 = np.sum(true_prob[0: int(min(num_pos, 1000)), 0]) / min(1000, num_pos)
    else:
        top10 = 0
        top100 = 0
        top500 = 0   
        top1000 = 0
    return {'AUROC': test_auroc, 'AUC': test_auroc, 'TPR': test_tpr, 'FPR': test_fpr,
            'AUPRC': test_aupr, 'precision': precision, 'recall': recall,
            'AUPR': test_aupr, 
            'AP': test_ap, 
            'F1': F1,
            'F1_cutoff': cutoff,
            'balanced_ACC': test_balanced_acc,
            'pos-neg': biock.label_count(true),
            'top10': top10, 'top100': top100, 'top500': top500, 'top1000': top1000,
            'true': true,
            'prob': prob}


def draw_loss_auc(train_loss, train_auc, validate_loss=list(), validate_auc=list(), save_name=None, show=False):
    if len(train_loss) < 2:
        return None
    max_len = max(len(train_loss), len(validate_loss), len(train_auc), len(validate_auc))
    y_ticks = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(min(max(8, max_len), np.sqrt(50 * max_len).astype(int)), 16))
    ax1 = plt.subplot(211)
    plt.plot(train_loss, 'b:', label="Train loss")
    plt.plot(validate_loss, 'r', label="Validate loss")
    plt.xticks(range(max_len))
    #plt.yticks(y_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.grid(color='grey', linestyle=':')
    plt.legend()

    ax2 = plt.subplot(212)
    plt.plot(train_auc, 'b:', label="Train AUC")
    plt.plot(validate_auc, 'r', label="Validate AUC")
    plt.xticks(range(max_len))
    plt.yticks(y_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(color='grey', linestyle=':')
    plt.legend()

    if save_name is not None:
        if save_name.endswith("pdf"):
            vector = save_name
            bit = save_name.rstrip('pdf') + "png"
        else:
            vector = save_name + ".pdf"
            bit = save_name
        plt.savefig(vector)
        plt.savefig(bit, dpi=300)
    if show:
        plt.show()
    plt.close()

def draw_cv(fold_auc, fold_aupr, all_auc, all_aupr, save_name=None, show=False):

    lw=1.5
    fold_colors = ['g', 'c', 'm', 'y', 'grey']
    max_len = len(fold_auc)
    fold_auc_ar = np.array(fold_auc)
    fold_aupr_ar = np.array(fold_aupr)
    auc_err = fold_auc_ar.std(axis=1)
    aupr_err = fold_aupr_ar.std(axis=1)
    x = list(range(len(fold_auc)))
    x = [i + 1 for i in x]
    y_ticks = np.arange(0, 1.1, 0.1)

    plt.figure(figsize=(min(max(8, max_len), np.sqrt(50 * max_len).astype(int)), 16))
    # auc
    ax1 = plt.subplot(211)
    plt.errorbar(x, all_auc, yerr=auc_err, color='r', label="AUC", marker='x', linewidth=lw)
    for i in range(fold_auc_ar.shape[1]):
        plt.plot(x, fold_auc_ar.T[i], c=fold_colors[i], linewidth=1, label="fold-{}".format(i + 1), linestyle='dashed')
    plt.xticks(x)
    plt.yticks(y_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("CV AUC")
    plt.grid(color='grey', linestyle=':')
    plt.legend()

    ax2 = plt.subplot(212)
    plt.errorbar(x, all_aupr, yerr=aupr_err, color='b', label="AUPR", marker='x', linewidth=lw)
    for i in range(fold_aupr_ar.shape[1]):
        plt.plot(x, fold_aupr_ar.T[i], c=fold_colors[i], linewidth=1, label="fold-{}".format(i + 1), linestyle='dashed')
    plt.xticks(x)
    plt.yticks(y_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("CV AUPR")
    plt.grid(color='grey', linestyle=':')
    plt.legend()

    if save_name is not None:
        if save_name.endswith("pdf"):
            vector = save_name
            bit = save_name.rstrip('pdf') + "png"
        else:
            vector = save_name + ".pdf"
            bit = save_name
        plt.savefig(vector)
        plt.savefig(bit, dpi=300)
    if show:
        plt.show()
    plt.close()

def resplit_train_test(dataset, chroms, test_chroms=['chrX', 'chr20', 'chr21', 'chr22']):
    print("  - Splitting dataset ...")
    if type(test_chroms) is str:
        test_chroms = [test_chroms]
    elif type(test_chroms) is set:
        test_chroms = list(test_chroms)
    chroms = chroms.squeeze()
    indexes = np.array(list(range(len(chroms))))

    test_bool = np.isin(chroms, test_chroms)
    test_idx = indexes[test_bool]

    train_bool = np.logical_not(test_bool)
    train_idx = indexes[train_bool]

    train_data, test_data = dict(), dict()
    keys = list(dataset.keys())
    for k in keys:
        try:
            if dataset[k] is None:
                train_data[k] = None
                test_data[k] = None
            else:
                train_data[k] = dataset[k][train_idx]
                test_data[k] = dataset[k][test_idx]
        except TypeError as err:
            print("    Skip key: '{}' ({})".format(k, err))
        del dataset[k]
    del dataset
    print("  - Train chroms: {}".format(biock.label_count(train_data['chrom'])))
    print("    Validate chroms: {}".format(biock.label_count(test_data['chrom'])))
    return train_data, test_data

class DeterminedGroupKFold(object):
    def __init__(self, n_splits=5):
        pass

    def split(self, X, groups, folds, y=None):
        assert type(folds) is dict
        indexes = np.array(list(range(X.shape[0])))
        groups = np.array(groups).squeeze()
        fold_id = sorted(folds.keys())
        for idx in fold_id:
            is_in = np.isin(groups, list(folds[idx]))
            not_in = np.logical_not(is_in)

            yield indexes[not_in], indexes[is_in]
