#!/usr/bin/env python3

import argparse, os, sys
import numpy as np

from variables import *


def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input')
    p.add_argument('-o', '--outdir', required=True)
    p.add_argument('--bed-config', nargs='+')
    p.add_argument('--buildver', default="hg19", choices=('hg19', 'hg38'))
    p.add_argument('--bw-config', nargs='+', default=None)
    p.add_argument('--interval-len', default=3000000, type=int)
    # p.add_argument('--bin0size', default=1000, type=int)
    #p.add_argument('--seed', type=int, default=2020)
    return p.parse_args()

available_models = {'epi': "/home/chenken/Documents/DeepEPI/models/OCT29_5_features.model"}


if __name__ == "__main__":
    args = get_args()
    #np.random.seed(args.seed)
    chroms, enhancers, promoters = list(), list(), list()
    with open(args.input) as infile:
        for l in infile:
            if l.startswith("##"):
                if l.startswith("##CELLTYPE"):
                    celltype = l.strip().split()[1]
                elif l.startswith("##MODEL"):
                    model_name = l.strip().split()[1]
                    assert model_name.lower() in available_models.keys()
            elif l.startswith("#"):
                continue
            elif len(l.strip()) > 0:
                chrom, enhancer, promoter = l.split()[0:3]
                if '-' in enhancer:
                    e1, e2 = enhancer.split('-')
                    enhancer = (int(e1) + int(e2))//2
                if '-' in promoter:
                    p1, p2 = promoter.split('-')
                    promoter = (int(p1) + int(p2))//2
                assert enhancer > 0 and promoter > 0
                chroms.append(chrom)
                enhancers.append(enhancer)
                promoters.append(promoter)
    outdir = make_directory(args.outdir)
    with open("{}/{}_enhancer.bed".format(outdir, celltype), 'w') as out:
        for i, c in enumerate(chroms):
            out.write("{}\t{}\t{}\t{}||{}:{}-{}\n".format(c, enhancers[i], enhancers[i] + 1, celltype, c, enhancers[i], enhancers[i] + 1))
    with open("{}/{}_promoter.bed".format(outdir, celltype), 'w') as out:
        for i, c in enumerate(chroms):
            out.write("{}\t{}\t{}\t{}||{}:{}-{}\n".format(c, promoters[i], promoters[i] + 1, celltype, c, promoters[i], promoters[i] + 1))
    with open("{}/{}_segment.bed".format(outdir, celltype), 'w') as out:
        for i, c in enumerate(chroms):
            mid = (enhancers[i] + promoters[i]) // 2
            if mid - args.interval_len // 2 < 0:
                shift = args.interval_len - mid
            elif mid + args.interval_len // 2 >= chrom_size[args.buildver][c]:
                shift = chrom_size[args.buildver][c] - mid - args.interval_len // 2
            else:
                shift = 0
            sl = mid + shift - args.interval_len // 2
            sr = sl + args.interval_len
            out.write("{}\t{}\t{}\t{}|{}|{},{}\n".format(c, promoters[i], promoters[i] + 1, celltype, c, enhancers[i], promoters[i]))

