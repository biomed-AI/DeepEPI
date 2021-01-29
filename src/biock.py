#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip, logging, warnings
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, precision_recall_curve, auc
import functools

print = functools.partial(print, flush=True)
print_err = functools.partial(print, flush=True, file=sys.stderr)

### misc
def str2num(s):
    try:
        n = int(s)
    except:
        n = float(s)
    return n

def overlap_length(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 <= y2:
        length = min(x2, y2) - max(x1, y1)
    else:
        length = y2 - x1
    return length

def distance(x1, x2, y1, y2):
    """ interval distance """
    return -overlap_length(x1, x2, y1, y2)

def label_count(labels):
    """ labels should be list,np.array """
    categories, counts = np.unique(labels, return_counts=True)
    ratio = (counts / counts.sum()).round(3)
    return list(zip(categories, counts, ratio))


### logs
def print_run_info(args=None, out=sys.stdout):
    print("\n# PROG: '{}' started at {}".format(os.path.basename(sys.argv[0]), time.asctime()), file=out)
    print("## PWD: %s" % os.getcwd(), file=out)
    print("## CMD: %s" % ' '.join(sys.argv), file=out)
    if args is not None:
        print("## ARG: {}".format(args), file=out)

def prog_header(args=None, out=sys.stdout):
    print("\n# Started at {}".format(time.asctime()))
    print("## Command: {}".format(' '.join(sys.argv)), file=out)
    if args is not None:
        print("##: Args: {}".format(args))

def get_logger(log_level="INFO"):
    logging.basicConfig(
                format='[%(asctime)s %(levelname)s] %(message)s',
                    stream=sys.stdout
            )
    log = logging.getLogger(__name__)
    log.setLevel(log_level)
    return log


### file & directory
def dirname(fn):
    folder = os.path.basename(fn)
    if folder == '':
        folder = "./"
    return folder

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def run_bash(cmd):
    p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf8'), err.decode('utf8')
    rc = p.returncode
    return (rc, out, err)



def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


## evaluation
def aupr_score(true, prob):
    true, prob = np.array(true), np.array(prob)
    assert len(true.shape) == 1 or min(true.shape) == 1
    assert len(prob.shape) == 1 or min(prob.shape) == 1
    true, prob = true.reshape(-1), prob.reshape(-1)
    precision, recall, thresholds = precision_recall_curve(true, prob)
    aupr = auc(recall, precision)
    return aupr

class BasicBED(object):
    def __init__(self, input_file, bin_size=50000):
        self.input_file = input_file
        self.chroms = dict()
        self.bin_size = bin_size
        #self.parse_input()

    def intersect(self, chrom, start, end, gap=0):
        start, end = int(start) - gap, int(end) + gap
        if start >= end:
            warnings.warn("starat >= end: start={}, end={}".format(start, end))
        res = set()
        if chrom in self.chroms:
            for idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
                if idx not in self.chroms[chrom]:
                    continue
                try:
                    for i_start, i_end, attr in self.chroms[chrom][idx]:
                        if i_start >= end or i_end <= start:
                            continue
                        res.add((i_start, i_end, attr))
                except:
                    print(self.chroms[chrom][idx])
                    exit(1)
        res = sorted(list(res), key=lambda l:(l[0], l[1]))
        return res

    def sort(self, merge=False):
        for chrom in self.chroms:
            for idx in self.chroms[chrom]:
                self.chroms[chrom][idx] = \
                        sorted(self.chroms[chrom][idx], key=lambda l:(l[0], l[1]))

    def add_record(self, chrom, start, end, attrs=None, cut=False):
        if chrom not in self.chroms:
            self.chroms[chrom] = dict()
        for bin_idx in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
            if bin_idx not in self.chroms[chrom]:
                self.chroms[chrom][bin_idx] = list()
            if cut:
                raise NotImplementedError
            else:
                self.chroms[chrom][bin_idx].append((start, end, attrs))


    def __str__(self):
        return "BasicBED(filename:{})".format(os.path.relpath(self.input_file))

    def parse_input(self):
        raise NotImplementedError
        # record format: (left, right, (XXX))
        # XXX: self defined attributes of interval [left, right)

def array_summary(x):
    x = np.array(x)
    r = {'mean': np.mean(x).round(3), 'min': np.round(min(x), 3)}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        r[q] = np.quantile(x, q).round(3)
    r['max'] = np.round(max(x), 3)
    return r


## constants & variables
hg19_chromsize = {"chr1": 249250621, "chr2": 243199373, 
        "chr3": 198022430, "chr4": 191154276, 
        "chr5": 180915260, "chr6": 171115067, 
        "chr7": 159138663, "chr8": 146364022, 
        "chr9": 141213431, "chr10": 135534747, 
        "chr11": 135006516, "chr12": 133851895, 
        "chr13": 115169878, "chr14": 107349540, 
        "chr15": 102531392, "chr16": 90354753, 
        "chr17": 81195210, "chr18": 78077248, 
        "chr19": 59128983, "chr20": 63025520, 
        "chr21": 48129895, "chr22": 51304566, 
        "chrX": 155270560, "chrY": 59373566,
        "chrM": 16569, "chrMT": 16569}
chrom_size = {'hg19': hg19_chromsize, 'GRCh37': hg19_chromsize}

nt_onehot_dict = {
        'A': np.array([1, 0, 0, 0]), 'a': np.array([1, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0]), 'c': np.array([0, 1, 0, 0]),
        'G': np.array([0, 0, 1, 0]), 'g': np.array([0, 0, 1, 0]),
        'T': np.array([0, 0, 0, 1]), 't': np.array([0, 0, 0, 1]),
        'U': np.array([0, 0, 0, 1]), 'u': np.array([0, 0, 0, 1]),
        'W': np.array([0.5, 0, 0, 0.5]),
        'S': np.array([0, 0.5, 0.5, 0]),
        'N': np.array([0.25, 0.25, 0.25, 0.25]),
        'n': np.array([0.25, 0.25, 0.25, 0.25]), 
        ';': np.array([0, 0, 0, 0])
}


if __name__ == "__main__":
    #args = get_args()
    pass



#import argparse, os, sys, warnings, time
#import numpy as np
#
#def get_args():
#    p = argparse.ArgumentParser()
#    return p.parse_args()
#
#if __name__ == "__main__":
#    args = get_args()
#



