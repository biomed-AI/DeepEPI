#!/usr/bin/env python3

import argparse, functools, gzip, json, os, pickle, select, sys, shutil, time, tempfile, glob, random, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim, Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt

import EPI_misc
from idle_gpu import idle_gpu
from EPI_misc import max_indexes, min_indexes, evaluate_results, tensor2numpy
from EPIGL import EPIGL
import prepare_data

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


def print_run_info(args=None, out=sys.stdout):
    print("\n# PROG: '{}' started at {}".format(os.path.basename(sys.argv[0]), time.asctime()), file=out)
    print("## PWD: %s" % os.getcwd(), file=out)
    print("## CMD: %s" % ' '.join(sys.argv), file=out)
    if args is not None:
        print("## ARG: {}".format(args), file=out)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir



def predict(model, test_data, device, sample_idxs='all', batch_size=128, evaluation=True):
    model.eval()
    #model.to(device)
    criterion = nn.BCELoss()

    use_local, use_global = False, True

    if type(sample_idxs) is str and sample_idxs == 'all':
        sample_idxs = list(range(test_data['chrom'].shape[0]))
    total_size = len(sample_idxs)
    n_batches = np.ceil(total_size / batch_size).astype(int)
    with torch.no_grad():
        test_prob = None
        for i in range(n_batches):
            batch_sample_idxs = sample_idxs[i * batch_size : (i + 1) * batch_size]
            # label = test_data['label'][batch_sample_idxs]
            # label = Tensor(label).to(device)

            if model.use_dist:
                dist = validate_data['dist'][batch_sample_idxs]
                dist = Tensor(dist).to(device)
            else:
                dist = None
            
            enhancer, promoter = None, None

            segment = test_data['segment'][batch_sample_idxs]
            segment = Tensor(segment).to(device)

            prob = model(enhancer, promoter, segment, dist)

            if isinstance(prob, tuple):
                prob = prob[0]

            if test_prob is None:
                test_prob = prob
                # test_true = label
            else:
                test_prob = torch.cat((test_prob, prob), dim=0)
                # test_true = torch.cat((test_true, label), dim=0)
        # loss = criterion(test_prob, test_true).item()
    return test_prob.detach().cpu().numpy().squeeze()


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', "--data_dirs", required=True, nargs='+')
    p.add_argument('-m', "--model-state", required=True, help="Use model state dict")
    p.add_argument('-c', "--model-config", required=True, help="Model parameters")
    p.add_argument('-gf', "--global-features", default=["segment.mark_3", "segment.ep.pos-encode", "segment.CTCF.arcsinh", "segment.DNase-pval.arcsinh", "segment.H3K27ac-pval.arcsinh", "segment.H3K4me1-pval.arcsinh", "segment.H3K4me3-pval.arcsinh"], nargs='+', help="DeepEPI features")
    p.add_argument('-b', "--batch-size", default=128, type=int, help="batch size")
    p.add_argument('-o', '--output', help="Write prediction", required=True)
    p.add_argument("--seed", type=int, default=2020)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    print_run_info(args, out=sys.stderr)

    # deterministic
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_global, use_local = True, False

    ## Loat data
    print("- Loading dataset ...")
    validate_data = prepare_data.load_multiple_datasets(args.data_dirs, global_features=args.global_features)
    print("- Test data size: {}".format("segment {}".format(validate_data['segment'].shape) if use_global else ""))

    # gpu_id = idle_gpu(min_memory=3072)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("- Using device: ", device)

    params = json.load(open(args.model_config)) # {"description": xxx, "params": xxx}
    params['use_local'] = False
    params['use_global'] = True
    params['s_channel'] = validate_data['segment'].shape[1]
    print("- Model config: {}".format(params))

    model = EPIGL(**params)
    model.to(device)
    model.eval()

    print("- Loading model state from {}".format(args.model_state))
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_state)
    else:
        state_dict = torch.load(args.model_state, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])

    print("- Model\n", model, '\n', model_summary(model))

    results = predict(model, validate_data, device, sample_idxs='all', batch_size=args.batch_size, evaluation=True)
    print(results)
    np.savetxt("{}/results.txt".format(args.output), results, fmt="%.5f")

