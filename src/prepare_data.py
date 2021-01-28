#!/usr/bin/env python3

import argparse, functools, glob, gzip, json, os, pickle, sys, time, h5py, pyBigWig
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import GroupKFold, StratifiedKFold
from utils import peak_utils, wigFix_utils, retrieve_bigWig_signal

onehot_dict = {
        'A': np.array([1, 0, 0, 0]), 'a': np.array([1, 0, 0, 0]),
        'G': np.array([0, 1, 0, 0]), 'g': np.array([0, 1, 0, 0]),
        'C': np.array([0, 0, 1, 0]), 'c': np.array([0, 0, 1, 0]),
        'T': np.array([0, 0, 0, 1]), 't': np.array([0, 0, 0, 1]),
        'W': np.array([0.5, 0, 0, 0.5]),
        'S': np.array([0, 0.5, 0.5, 0]),
        'N': np.array([0.25, 0.25, 0.25, 0.25]),
        'n': np.array([0.25, 0.25, 0.25, 0.25]), 
        ';': np.array([0, 0, 0, 0])
}
def one_hot(seq, channel_first=True):
    onehot_seq = [onehot_dict[n] for n in seq] # (length, channel)
    onehot_seq = np.array(onehot_seq)
    if channel_first:
        onehot_seq = onehot_seq.T
    return  onehot_seq # (length, 4)


def prepare_metadata(name, data_dir):
    print("Processing metadata {} ... ({})".format(data_dir, time.asctime()))
    data_dir = data_dir.rstrip('/')
    save_name = "%s/%s_metadata.npz" % (data_dir.rstrip('/'), name)
    if os.path.exists(save_name):
        print("- {} exists, skipped".format(save_name))
        return None
    if os.path.exists("%s/%s_label.txt" % (data_dir, name)):
        labels = np.loadtxt("%s/%s_label.txt" % (data_dir, name)).reshape(-1, 1).astype(np.int)
    else:
        labels = None
    enhancer_df = pd.read_csv("%s/%s_enhancer.bed" % (data_dir, name), header=None, delimiter='\t')
    promoter_df = pd.read_csv("%s/%s_promoter.bed" % (data_dir, name), header=None, delimiter='\t')
    celltypes = list()
    with open("%s/%s_enhancer.bed" % (data_dir, name)) as infile:
        for l in infile:
            if l.startswith('#'):
                continue
            celltypes.append(l.split('\t')[3].split('|')[0])
    celltypes = np.array(celltypes).reshape(-1, 1)
    chroms = np.array(enhancer_df[0]).reshape(-1, 1)
    e_start = np.array(enhancer_df[1]).astype(np.int).reshape(-1, 1)
    e_end = np.array(enhancer_df[2]).astype(np.int).reshape(-1, 1)
    e_names = np.array(enhancer_df[3]).reshape(-1, 1)
    del enhancer_df
    p_start = np.array(promoter_df[1]).astype(np.int).reshape(-1, 1)
    p_end = np.array(promoter_df[2]).astype(np.int).reshape(-1, 1)
    p_names = np.array(promoter_df[3]).reshape(-1, 1)
    del promoter_df
    dist = (e_start + e_end - p_start - p_end) // 2
    dist = np.log10(1 + abs(dist) / 100000)
    up_down = (e_end < p_start).astype(np.int8)
    np.savez_compressed(save_name, celltype=celltypes, label=labels, chrom=chroms, e_name=e_names, p_name=p_names, e_start=e_start, e_end=e_end, p_start=p_start, p_end=p_end, up_down=up_down, dist=dist)
    print("- {} saved at {}".format(save_name, time.asctime()))


def prepare_sequence(name, data_dir):
    print("Processing sequence {} ... ({})".format(data_dir, time.asctime()))
    save_name_AGCT = "%s/%s_sequences.npz" % (data_dir.rstrip('/'), name)
    save_name_WS = "%s/%s_sequences.WS.npz" % (data_dir.rstrip('/'), name)
    if os.path.exists(save_name_AGCT) and os.path.exists(save_name_WS):
        print("- {} exists, skipped".format(save_name_AGCT))
        return None
    enhancers, promoters = list(), list()
    with gzip.open("%s/%s_enhancer.fa.gz" % (data_dir, name), 'rt') as infile:
        for l in infile:
            if l.startswith('#') or l.startswith('>'):
                continue
            enhancers.append(one_hot(l.strip()))
    enhancers = np.array(enhancers).astype(np.float16)
    with gzip.open("%s/%s_promoter.fa.gz" % (data_dir, name), 'rt') as infile:
        for l in infile:
            if l.startswith('#') or l.startswith('>'):
                continue
            promoters.append(one_hot(l.strip()))
    promoters = np.array(promoters).astype(np.float16)
    np.savez_compressed(save_name_AGCT, enhancer=enhancers, promoter=promoters)
    print("- {} saved. {}".format(save_name_AGCT, time.asctime()))
    enhancers_WS = np.concatenate((enhancers[:,[0,3],:].sum(axis=1, keepdims=True), 
                                   enhancers[:,[1,2],:].sum(axis=1, keepdims=True)),
                                   axis=1)
    promoters_WS = np.concatenate((promoters[:,[0,3],:].sum(axis=1, keepdims=True), 
                                   promoters[:,[1,2],:].sum(axis=1, keepdims=True)),
                                   axis=1)
    np.savez_compressed(save_name_WS, enhancer=enhancers_WS, promoter=promoters_WS)
    print("- {} saved. {}".format(save_name_AGCT, time.asctime()))
    print("- {} saved. {}".format(save_name_WS, time.asctime()))

def parse_segment_mark(s_starts, s_ends, e_mids, p_mids, bin_size=1000, bin_num=3000):
    samples_channels_length = list()
    for i, s_start in enumerate(s_starts):
        mark = [0 for i in range(bin_num)]
        e_mid, p_mid, s_end = e_mids[i], p_mids[i], s_ends[i]
        e_idx = (e_mid - s_start) // bin_size 
        p_idx = (p_mid - s_start) // bin_size
        mark[e_idx - 1], mark[e_idx], mark[e_idx + 1] = 1, 1, 1
        mark[p_idx - 1], mark[p_idx], mark[p_idx + 1] = 1, 1, 1
        mark = np.array(mark).reshape(1, -1)
        samples_channels_length.append(mark.astype(np.int8))
    samples_channels_length = np.array(samples_channels_length).astype(np.int8)
    return samples_channels_length

#def prepare_segment_dist(name, data_dir, CTCF_json=None, DNase_json=None, histone_json=None, nthreads=16, bin_size=1000, bin_num=3000, nthreads=16, batch_size=1200):
def prepare_segment_mark(name, data_dir, bin_size=1000, bin_num=3000, nthreads=16, batch_size=4800):
    print("Processing segment signals {} ... {}".format(data_dir, time.asctime()))
    save_name = "{}/{}_segment.mark_3.npz".format(data_dir, name)
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        #return None
    segment_bed = "{}/{}_segment.bed".format(data_dir, name)
    cells, s_starts, s_ends, e_mids, p_mids = list(), list(), list(), list(), list()
    with open(segment_bed) as infile:
        for l in infile:
            chrom, start, end, ID = l.strip().split()
            cell, _, e_mid_p_mid = ID.split('|')
            e_mid, p_mid = e_mid_p_mid.split(',')
            cells.append(cell)
            s_starts.append(int(start))
            s_ends.append(int(end))
            e_mids.append(int(e_mid))
            p_mids.append(int(p_mid))
    total = len(cells)
    batch_size = min(batch_size, total) 
    cnt = 0
    cre_mark = None
    while batch_size > 0:
        job_size = np.ceil(batch_size / nthreads).astype(int)
        n_jobs = np.ceil(batch_size / job_size).astype(int)
        job_list = [(s_starts[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_ends[cnt + i * job_size : cnt + (i + 1) * job_size], 
            e_mids[cnt + i * job_size : cnt + (i + 1) * job_size], 
            p_mids[cnt + i * job_size : cnt + (i + 1) * job_size], 
            bin_size, bin_num) for i in range(n_jobs)]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.starmap(parse_segment_mark, job_list)
        if cre_mark is None:
            cre_mark = np.concatenate(res, axis=0)
        else:
            cre_mark = np.concatenate(([cre_mark] + res), axis=0)
        cnt += batch_size
        batch_size = min(batch_size, total - cnt)
        assert batch_size >= 0
    np.savez_compressed(save_name, segment=cre_mark)
    print("- {} {} {} saved. {}".format(save_name, cre_mark.shape, (cre_mark.max(axis=2).mean(axis=0), cre_mark.min(axis=2).mean(axis=0)), time.asctime()))



def sym_log(x):
    if type(x) is float or type(x) is int:
        x = np.array([x])
    elif type(x) is list:
        x = np.array(x)
    sign = np.where(x < 0, -1, 1)
    return sign * np.log(abs(x) + 1)

def parse_segment_dist(s_starts, s_ends, e_mids, p_mids, bin_size=1000, bin_num=3000, k=100, two_channel=False):
    samples_channels_length = list()
    for i, s_start in enumerate(s_starts):
        if two_channel:
            dist_e, dist_p = list(), list()
        else:
            dist = [0 for i in range(bin_num)]
        e_mid, p_mid, s_end = e_mids[i], p_mids[i], s_ends[i]
        e_idx = (e_mid - s_start) // bin_size
        p_idx = (p_mid - s_start) // bin_size
        if two_channel:
            for idx in range(bin_num):
                #dist_e.append(abs(idx - e_idx))
                #dist_p.append(abs(idx - p_idx))
                dist_e.append(idx - e_idx)
                dist_p.append(idx - p_idx)
            if e_mid < p_mid:
                dist_p = -np.array(dist_p)
            else:
                dist_e = -np.array(dist_e)
            dist = np.array([np.array(dist_e), np.array(dist_p)])
        else:
            for idx in range(bin_num):
                dist[idx] = min(abs(idx - e_idx), abs(idx - p_idx))
            dist = np.array(dist).reshape(1, -1)
        #dist = k / (k + dist)
        dist = sym_log(dist / 1000)
        samples_channels_length.append(dist)
    samples_channels_length = np.array(samples_channels_length).astype(np.float16)
    return samples_channels_length

#def prepare_segment_dist(name, data_dir, CTCF_json=None, DNase_json=None, histone_json=None, nthreads=16, bin_size=1000, bin_num=3000, nthreads=16, batch_size=1200):
def prepare_segment_dist(name, data_dir, bin_size=1000, bin_num=3000, nthreads=16, batch_size=4800, dist_k=100, two_channel=False):
    print("Processing segment signals {} ... {}".format(data_dir, time.asctime()))
    save_name = "{}/{}_segment.{}pos-encode.npz".format(data_dir, name, "ep." if two_channel else "")
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        return None
    segment_bed = "{}/{}_segment.bed".format(data_dir, name)
    cells, s_starts, s_ends, e_mids, p_mids = list(), list(), list(), list(), list()
    with open(segment_bed) as infile:
        for l in infile:
            chrom, start, end, ID = l.strip().split()
            cell, _, e_mid_p_mid = ID.split('|')
            e_mid, p_mid = e_mid_p_mid.split(',')
            cells.append(cell)
            s_starts.append(int(start))
            s_ends.append(int(end))
            e_mids.append(int(e_mid))
            p_mids.append(int(p_mid))
    total = len(cells)
    batch_size = min(batch_size, total) 
    cnt = 0
    cre_dist = None
    while batch_size > 0:
        job_size = np.ceil(batch_size / nthreads).astype(int)
        n_jobs = np.ceil(batch_size / job_size).astype(int)
        job_list = [(s_starts[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_ends[cnt + i * job_size : cnt + (i + 1) * job_size], 
            e_mids[cnt + i * job_size : cnt + (i + 1) * job_size], 
            p_mids[cnt + i * job_size : cnt + (i + 1) * job_size], 
            bin_size, bin_num, dist_k, two_channel) for i in range(n_jobs)]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.starmap(parse_segment_dist, job_list)
        if cre_dist is None:
            cre_dist = np.concatenate(res, axis=0)
        else:
            cre_dist = np.concatenate(([cre_dist] + res), axis=0)
        cnt += batch_size
        batch_size = min(batch_size, total - cnt)
        assert batch_size >= 0
    np.savez_compressed(save_name, segment=cre_dist)
    print("- {} {} {} saved. {}".format(save_name, cre_dist.shape, (cre_dist.max(axis=2).mean(axis=0), cre_dist.min(axis=2).mean(axis=0)), time.asctime()))


def parse_bed_peaks(celltypes, chroms, starts, ends, peaks, assays, pool_size=1000):
    samples_channels_length = list()
    bin_num = (ends[0] - starts[0]) // pool_size
    for i, ct in enumerate(celltypes): # for each sample
        chrom, start, end = chroms[i], starts[i], ends[i]
        channels_length = list()
        for a in assays:
            peak_vals = np.array(peaks[ct][a].query_region(chrom, start, end, to_array=True, norm_method="raw"))
            if pool_size > 1:
                ar = [np.mean(peak_vals[i * pool_size : (i + 1) * pool_size]) for i in range(bin_num)]
                del peak_vals
                peak_vals = np.array(ar)
            channels_length.append(ar)
        samples_channels_length.append(np.array(channels_length).astype(np.float16))
    samples_channels_length = np.array(samples_channels_length).astype(np.float16)
    return samples_channels_length

def prepare_segment_bed_peak(name, data_dir, bed_config, bin_size=1000, bin_num=3000, nthreads=16, batch_size=4800):
    print("Processing segment bed {} ... {}".format(data_dir, time.asctime()))
    config = json.load(open(bed_config))
    dbname = config['dbname']
    save_name = "{}/{}_segment.{}.arcsinh.npz".format(data_dir.rstrip('/'), name, dbname)
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        return None
    segment_bed = "{}/{}_segment.bed".format(data_dir, name)
    cells, chroms, s_starts, s_ends = list(), list(), list(), list() 
    with open(segment_bed) as infile:
        for l in infile:
            chrom, start, end, ID = l.strip().split()
            cell, _, e_mid_p_mid = ID.split('|')
            e_mid, p_mid = e_mid_p_mid.split(',')
            cells.append(cell)
            chroms.append(chrom)
            s_starts.append(int(start))
            s_ends.append(int(end))
    peaks, assays, _, _, dbname = peak_utils.load_peaks(bed_config)
    total = len(cells)
    batch_size = min(batch_size, total) 
    cnt = 0
    bed_peaks = None
    while batch_size > 0:
        job_size = np.ceil(batch_size / nthreads).astype(int)
        n_jobs = np.ceil(batch_size / job_size).astype(int)
        #print([(i, n_jobs, cnt + i * job_size, cnt + (i + 1) * job_size) for i in range(n_jobs)])
        job_list = [(cells[cnt + i * job_size : cnt + (i + 1) * job_size], 
            chroms[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_starts[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_ends[cnt + i * job_size : cnt + (i + 1) * job_size], 
            peaks, assays, bin_size) for i in range(n_jobs)]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.starmap(parse_bed_peaks, job_list)
        if bed_peaks is None:
            bed_peaks = np.concatenate(res, axis=0)
        else:
            bed_peaks = np.concatenate(([bed_peaks] + res), axis=0)
        cnt += batch_size
        print("    {}/{} finished {} ({})".format(cnt, total, bed_peaks.shape, time.asctime()))
        batch_size = min(batch_size, total - cnt)
        assert batch_size >= 0
    np.savez_compressed(save_name, segment=np.arcsinh(bed_peaks))
    print("- {} {} saved. {}".format(save_name, bed_peaks.shape, time.asctime()))



def parse_bw_signals(celltypes, chroms, starts, ends, bigwig_config, pool_size=1000):
    config = json.load(open(bigwig_config))
    assays = config['selected']
    bin_num = (ends[0] - starts[0]) // pool_size
    samples_channels_length = list()
    for i, ct in enumerate(celltypes):
        chrom, start, end = chroms[i], starts[i], ends[i]
        channels_length = list()
        for a in assays:
            bigwig = os.path.join(config['location'], config['celltypes'][ct][a])
            bw = pyBigWig.open(bigwig)
            signals = np.array(bw.values(chrom, start, end))#.reshape(1, -1)
            bw.close()
            signals = np.nan_to_num(signals).astype(np.float16)
            if pool_size > 1:
                ar = [np.mean(signals[i * pool_size : (i + 1) * pool_size]) for i in range(bin_num)]
                del signals
                signals = np.array(ar)
            channels_length.append(signals)
        samples_channels_length.append(channels_length)
    samples_channels_length = np.array(samples_channels_length).astype(np.float16)
    return samples_channels_length

def prepare_segment_bw_signals(name, data_dir, bw_config, bin_size=1000, bin_num=3000, nthreads=16, batch_size=4800):
    print("Processing segment bigwig {} ... {}".format(data_dir, time.asctime()))
    config = json.load(open(bw_config))
    dbname = config['dbname']
    save_name = "{}/{}_segment.{}.arcsinh.npz".format(data_dir.rstrip('/'), name, dbname)
    if os.path.exists(save_name):
        print("- {} exists, skipped.".format(save_name))
        return None
    segment_bed = "{}/{}_segment.bed".format(data_dir, name)
    cells, chroms, s_starts, s_ends = list(), list(), list(), list() 
    with open(segment_bed) as infile:
        for l in infile:
            chrom, start, end, ID = l.strip().split()
            cell, _, e_mid_p_mid = ID.split('|')
            e_mid, p_mid = e_mid_p_mid.split(',')
            cells.append(cell)
            chroms.append(chrom)
            s_starts.append(int(start))
            s_ends.append(int(end))
    total = len(cells)
    batch_size = min(batch_size, total) 
    cnt = 0
    bw_signals = None
    while batch_size > 0:
        job_size = np.ceil(batch_size / nthreads).astype(int)
        n_jobs = np.ceil(batch_size / job_size).astype(int)
        #print([(i, n_jobs, cnt + i * job_size, cnt + (i + 1) * job_size) for i in range(n_jobs)])
        job_list = [(cells[cnt + i * job_size : cnt + (i + 1) * job_size], 
            chroms[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_starts[cnt + i * job_size : cnt + (i + 1) * job_size], 
            s_ends[cnt + i * job_size : cnt + (i + 1) * job_size], 
            bw_config, bin_size) for i in range(n_jobs)]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.starmap(parse_bw_signals, job_list)
        if bw_signals is None:
            bw_signals = np.concatenate(res, axis=0)
        else:
            bw_signals = np.concatenate(([bw_signals] + res), axis=0)
        cnt += batch_size
        print("    {}/{} finished {} ({})".format(cnt, total, bw_signals.shape, time.asctime()))
        batch_size = min(batch_size, total - cnt)
        assert batch_size >= 0
    np.savez_compressed(save_name, segment=np.arcsinh(bw_signals))
    print("- {} {} saved. {}".format(save_name, bw_signals.shape, time.asctime()))


def load_dataset(data_dir, global_features=None, min_memory_GB=4):
    import psutil
    BytePerGB = 1073741824
    time_cnt = 0
    while psutil.virtual_memory().available < min_memory_GB * BytePerGB:
        if time_cnt % 600 == 0:
            print("* Waiting for memory ... {}".format(time.asctime()))
        time.sleep(120)
        time_cnt += 120
    if psutil.virtual_memory().available / BytePerGB < 32:
        print("* Warning: available memory is less than 32GB ({:.2f}GB)".format(psutil.virtual_memory().available / BytePerGB))

    data_dir = data_dir.rstrip('/')
    try:
        fn = glob.glob("%s/*_metadata.npz" % data_dir)[0]
    except:
        print("* ERROR: metadata not found under {}".format(data_dir))
        exit(1)
    name = fn.split('/')[-1].split('_')[0]
    local_features =  list() 
    metadata = np.load("%s/%s_metadata.npz" % (data_dir, name), allow_pickle=True)
    # label = metadata['label'].astype(np.int8).reshape(-1, 1) if metadata['label'] is not None else None
    chrom = metadata['chrom'].reshape(-1, 1)
    dist = metadata['dist'].reshape(-1, 1)
    up_down = metadata['up_down'].reshape(-1)
    enhancer, promoter, segment = None, None, None
    if global_features is not None:
        for i, fn in enumerate(global_features):
            print("{}/{}_{}.npz".format(data_dir, name, fn))
            s = np.load("{}/{}_{}.npz".format(data_dir, name, fn))['segment'].astype(np.float16)
            if len(s.shape) == 2:
                nr, nc = s.shape
                s = s.reshape(nr, 1, nc)
            if i == 0:
                segment = s
            else:
                segment = np.concatenate((segment, s), axis=1)
    dataset = {"label": label, "dist": dist,
            "chrom": chrom, "up_down": up_down,
            "enhancer": enhancer, 
            "promoter": promoter, 
            "segment": segment}
    return dataset


def load_multiple_datasets(data_dirs, global_features=None):
    shape = {'enhancer': None, 'promoter': None, 'segment': None}
    for i, data_dir in enumerate(data_dirs):                                       
        assert os.path.isdir(data_dir), "ERROR: '%s' doesn't exist !" % data_dir
        print("  Loading dataset from {} ...".format(data_dir), end=' ')
        if i == 0:
            dataset = load_dataset(data_dir, global_features=global_features)
        else:
            tmp_dataset = load_dataset(data_dir, global_features=global_features)
            if global_features is not None:
                dataset['segment'] = np.concatenate((dataset['segment'], tmp_dataset['segment']), axis=0)
                del tmp_dataset['segment']
                shape['segment'] = dataset['segment'].shape
            dataset['chrom'] = np.concatenate((dataset['chrom'], tmp_dataset['chrom']), axis=0)
            dataset['dist'] = np.concatenate((dataset['dist'], tmp_dataset['dist']), axis=0)
            # dataset['label'] = np.concatenate((dataset['label'], tmp_dataset['label']), axis=0)
            dataset['up_down'] = np.concatenate((dataset['up_down'], tmp_dataset['up_down']))
            keys = set(tmp_dataset.keys())
            for k in keys:
                del tmp_dataset[k]
            del tmp_dataset
        dataset['shape'] = shape
        print("Done at {}".format(time.asctime()))
    return dataset


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', "--data_dir", required=True)
    p.add_argument('--name', required=True, help="Name")
    p.add_argument('--metadata', action='store_true')
    p.add_argument('--segment-bigwig', nargs='+', help="bigwig config for segment")
    p.add_argument('--segment-bed', nargs='+', help="bed config for segment")
    p.add_argument('--segment-mark', action='store_true', help="Mark segment")
    p.add_argument('--segment-dist', action='store_true', help="segment dist encoding")
    p.add_argument('-t', "--nthreads", default=16, type=int)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.metadata:
        prepare_metadata(args.name, args.data_dir)
    if args.segment_mark:
        prepare_segment_mark(args.name, args.data_dir, bin_size=1000, bin_num=3000, nthreads=args.nthreads)
    if args.segment_dist:
        prepare_segment_dist(args.name, args.data_dir, bin_size=1000, bin_num=3000, nthreads=args.nthreads, two_channel=True)
    if args.segment_bed is not None:
        for bed_config in args.segment_bed:
            prepare_segment_bed_peak(args.name, args.data_dir, bed_config, bin_size=1000, bin_num=3000, nthreads=args.nthreads)
    if args.segment_bigwig is not None:
        for bw_config in args.segment_bigwig:
            prepare_segment_bw_signals(args.name, args.data_dir, bw_config, bin_size=1000, bin_num=3000, nthreads=args.nthreads)
