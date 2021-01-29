#!/usr/bin/env python3

import argparse, functools, gzip, json, os, pickle, select, sys, shutil, time, tempfile, glob, random, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim, Tensor
from torch.autograd import Variable
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

import EPI_misc
from biock.idle_gpu import idle_gpu
from EPI_misc import max_indexes, min_indexes, evaluate_results, tensor2numpy, draw_loss_auc
from EPIGL import EPIGL
import prepare_data

from biock import print


def flip_DNA_feature(ar, reverse_intervals=None, require_flip=False):
    """ ar: (batch_size, channels, length) """
    if not require_flip:
        if reverse_intervals is not None:
            warnings.warn("require_flip=False but reverse_intervals is not None!")
        return ar
    elif reverse_intervals is None or len(reverse_intervals) == 0:
        return ar[:,:,::-1]
    else:
        new_idx = list(range(ar.shape[1]))
        flip_dict = dict()
        for a, b in reverse_intervals:
            for i in range(a, b):
                flip_dict[i] = (a, b)
        for i in new_idx.copy():
            if i in flip_dict:
                new_idx[i] = flip_dict[i][1] - 1 - (i - flip_dict[i][0])
        return ar[:, new_idx, ::-1]


def train_model(model, optimizer, train_data, validate_data=None, sample_idxs='all', checkpoint=None, batch_size=128, att_C=0.1, epoch=200, freeze_CNN=False, freeze_LSTM=False, freeze_ATT=False, shuffle=True, device=torch.device('cuda'), learning_curve=None):
    if type(sample_idxs) is str and sample_idxs == 'all':
        sample_idxs = list(range(train_data['label'].shape[0]))
    if shuffle:
        np.random.shuffle(sample_idxs)
        # for evaluate the result on train data set
        subsample_idx = sample_idxs.copy()
        np.random.shuffle(subsample_idx)
        subsample_idx = subsample_idx[0:min(len(sample_idxs), 8192)]

    use_local, use_global = False, False
    if 'enhancer' in train_data and train_data['enhancer'] is not None:
        use_local = True
    if 'segment' in train_data and train_data['segment'] is not None:
        use_global = True

    criterion = nn.BCELoss()

    total_size = len(sample_idxs)
    n_batches = total_size // batch_size # XXX: fixed batch_size

    train_loss, train_auc, train_aupr = list(), list(), list()
    validate_loss, validate_auc, validate_aupr = list(), list(), list()


    for epoch_idx in range(epoch):
        model.train()
        for i in range(n_batches):
            batch_sample_idxs = sample_idxs[i * batch_size : (i + 1) * batch_size]
            up_down = train_data['up_down'][batch_sample_idxs]
            label = train_data['label'][batch_sample_idxs]

            # flip
            up_down_idx, down_up_idx  = np.where(up_down == 1)[0], np.where(up_down == 0)[0]
            label = np.concatenate(
                    (label[up_down_idx], label[down_up_idx]), axis=0)
            label = Tensor(label).to(device)

            if model.use_dist:
                dist = train_data['dist'][batch_sample_idxs]
                dist = np.concatenate(
                        (dist[up_down_idx], dist[down_up_idx]), axis=0)
                dist = Tensor(dist).to(device)
            else:
                dist = None

            if use_local:
                enhancer = train_data['enhancer'][batch_sample_idxs]
                promoter = train_data['promoter'][batch_sample_idxs]

                reversed_enhancer = flip_DNA_feature(enhancer[down_up_idx], reverse_intervals=reverse_intervals, require_flip=require_flip)
                reversed_promoter = flip_DNA_feature(promoter[down_up_idx], reverse_intervals=reverse_intervals, require_flip=require_flip)
                enhancer = np.concatenate(
                        (enhancer[up_down_idx], reversed_enhancer), axis=0)
                promoter = np.concatenate(
                        (promoter[up_down_idx], reversed_promoter), axis=0)

                promoter = Tensor(promoter).to(device)
                enhancer = Tensor(enhancer).to(device)
            else:
                enhancer, promoter = None, None
            if use_global:
                segment = train_data['segment'][batch_sample_idxs]
                segment = np.concatenate(
                        (
                            segment[up_down_idx],
                            segment[down_up_idx]
                            #flip_DNA_feature(segment[down_up_idx], require_flip=False, reverse_intervals=None)
                        ), axis=0)
                segment = Tensor(segment).to(device)
            else:
                segment = None

            # train global module
            if use_global:
                model.freeze_local(cnn=freeze_CNN, lstm=freeze_LSTM, att=freeze_LSTM)
                model.unfreeze_global(cnn=not freeze_CNN, lstm=not freeze_LSTM, att=not freeze_ATT)
                prob, _, att = model(enhancer, promoter, segment, dist)
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(label.size(0), att.size(1), att.size(1)))
                penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)
                loss = criterion(prob, label) + (att_C * penal / label.size(0)).type(torch.cuda.FloatTensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del penal, att, attT, identity

            # train local module
            if use_local:
                model.freeze_global(cnn=freeze_CNN, lstm=freeze_LSTM, att=freeze_LSTM)
                model.unfreeze_local(cnn=not freeze_CNN, lstm=not freeze_LSTM, att=not freeze_ATT)
                prob, att, _ = model(enhancer, promoter, segment, dist)
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(label.size(0), att.size(1), att.size(1)))
                penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)
                loss = criterion(prob, label) + (att_C * penal / label.size(0)).type(torch.cuda.FloatTensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del penal, att, attT, identity

            if (i + 1) % (n_batches // 3) == 0:
                print("  TRAIN-Epoch [{}/{}], Step [{}/{}], loss={:.5f}".format(epoch_idx + 1, epoch, i + 1, n_batches, loss.item()))

        model.trained_epoch += 1
        model.unfreeze_local()
        model.unfreeze_global()
        if checkpoint:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, "%s.%02d" % (checkpoint, epoch_idx + 1))

        ## BEGIN: Evaluation
        model.eval()
        sampled_results = test_model(model, train_data, device, sample_idxs=subsample_idx, evaluation=True)
        print("  Train of epoch {: 3d}: loss={:.5f} AUC={:.5f} AUPR={:.5f}".format(model.trained_epoch, sampled_results['loss'], sampled_results['AUROC'], sampled_results['AUPR']))
        train_loss.append(sampled_results['loss'])
        train_auc.append(sampled_results['AUC'])
        train_aupr.append(sampled_results['AUPR'])

        if validate_data:
            validate_results = test_model(model, validate_data, device, batch_size=batch_size)
            validate_loss.append(validate_results['loss'])
            validate_auc.append(validate_results['AUROC'])
            validate_aupr.append(validate_results['AUPR'])
            print("* Test  of epoch {: 3d}: loss={:.5f} AUC={:.5f} AUPR={:.5f}  {:s}".format(model.trained_epoch, validate_loss[-1], validate_auc[-1], validate_aupr[-1], time.asctime()))
        # draw learning curve
        draw_loss_auc(train_loss, train_auc, validate_loss, validate_auc, save_name=learning_curve)
        if learning_curve is not None:
            learning_curve_data = '.'.join(learning_curve.split('.')[0:-1]) + ".pkl"
            with open(learning_curve_data, 'wb') as handle:
                pickle.dump({'train_loss': train_loss, \
                        'train_auc': train_auc, \
                        'train_aupr': train_aupr, \
                        'validate_loss': validate_loss, \
                        'validate_auc': validate_auc, \
                        'validate_aupr': validate_aupr, \
                        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print()

    return model


def test_model(model, test_data, device, sample_idxs='all', batch_size=128, evaluation=True):
    model.eval()
    #model.to(device)
    criterion = nn.BCELoss()

    use_local, use_global = False, False
    if 'enhancer' in test_data and test_data['enhancer'] is not None:
        use_local = True
    if 'segment' in test_data and test_data['segment'] is not None:
        use_global = True

    if type(sample_idxs) is str and sample_idxs == 'all':
        sample_idxs = list(range(test_data['label'].shape[0]))
    total_size = len(sample_idxs)
    n_batches = np.ceil(total_size / batch_size).astype(int)
    with torch.no_grad():
        test_prob, test_true = None, None
        for i in range(n_batches):
            batch_sample_idxs = sample_idxs[i * batch_size : (i + 1) * batch_size]
            up_down = test_data['up_down'][batch_sample_idxs]
            up_down_idx, down_up_idx  = np.where(up_down == 1)[0], np.where(up_down == 0)[0]
            label = test_data['label'][batch_sample_idxs]
            label = np.concatenate(
                    (label[up_down_idx], label[down_up_idx]), axis=0)
            label = Tensor(label).to(device)

            if model.use_dist:
                dist = train_data['dist'][batch_sample_idxs]
                dist = np.concatenate(
                        (dist[up_down_idx], dist[down_up_idx]), axis=0)
                dist = Tensor(dist).to(device)
            else:
                dist = None
            
            if use_local:
                enhancer = test_data['enhancer'][batch_sample_idxs]
                promoter = test_data['promoter'][batch_sample_idxs]

                reversed_enhancer = flip_DNA_feature(enhancer[down_up_idx], reverse_intervals=reverse_intervals, require_flip=require_flip)
                reversed_promoter = flip_DNA_feature(promoter[down_up_idx], reverse_intervals=reverse_intervals, require_flip=require_flip)
                enhancer = np.concatenate(
                        (enhancer[up_down_idx], reversed_enhancer), axis=0)
                promoter = np.concatenate(
                        (promoter[up_down_idx], reversed_promoter), axis=0)

                enhancer = Tensor(enhancer).to(device)
                promoter = Tensor(promoter).to(device)
            else:
                enhancer, promoter = None, None

            if use_global:
                segment = test_data['segment'][batch_sample_idxs]
                segment = np.concatenate(
                        (
                            segment[up_down_idx], 
                            flip_DNA_feature(segment[down_up_idx], require_flip=True, reverse_intervals=None)
                            ), axis=0)
                segment = Tensor(segment).to(device)
            else:
                segment = None

            prob = model(enhancer, promoter, segment, dist)

            if isinstance(prob, tuple):
                prob = prob[0]

            if test_prob is None:
                test_prob = prob
                test_true = label
            else:
                test_prob = torch.cat((test_prob, prob), dim=0)
                test_true = torch.cat((test_true, label), dim=0)
        loss = criterion(test_prob, test_true).item()
    if evaluation:
        results = evaluate_results(test_true, test_prob)
    else:
        results = dict()
    results['loss'] = loss
    results['prob'] = tensor2numpy(test_prob).reshape(-1, 1)
    return results


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--train-dirs', required=True, nargs='+', help="Directories containing data for pre-training")
    p.add_argument('-t', "--validate-chrs", nargs='+', default=['chr20', 'chr21', 'chr22', 'chrX'])
    p.add_argument('-o', "--outdir", required=True, help="Output directory")
    p.add_argument('--optim', choices=('SGD', 'ADAM'), default='ADAM')
    p.add_argument('--opt', choices=('SGD', 'ADAM'), default=None)
    p.add_argument('-w', "--weight-decay", default=0, type=float)
    p.add_argument('-m', "--model-state", default=None, help="Use model state dict")
    p.add_argument('-p', "--optimizer-state", default=None, help="Use optimizer state dict")
    p.add_argument('-l', "--learning-rate", default=1e-6, required=True, type=float, help="Learning rate (local)")
    p.add_argument('-n', "--model-name", default="EPIGL", help="Prefix of saved model")
    p.add_argument('-c', "--model-config", required=True, help="Model parameters")
    p.add_argument('--freeze-CNN', action='store_true')
    p.add_argument('--freeze-LSTM', action='store_true')
    p.add_argument('--freeze-ATT', action='store_true')
    p.add_argument('-lf', "--local-features", nargs='+', default=None, help="Additional feature middle names")
    p.add_argument('-gf', "--global-features", default=None, nargs='+', help="Surrounding features")
    p.add_argument('--epoch', type=int, required=True, help="Max epoch number")
    p.add_argument("--att-C", default=0.1, type=float, help="penal for l2 regularization")
    p.add_argument('-b', "--batch-size", default=128, type=int, help="batch size")
    p.add_argument("--seed", type=int, default=2020)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    biock.print_run_info(args)

    require_flip = True
    reverse_intervals = [(0, 4), (4, 6)]
    if require_flip:
        print("\n============= require flip ===============")
        print("reverse intervals {}".format(reverse_intervals))
        print("==========================================")
    
    outdir = biock.make_directory(args.outdir.rstrip('/'))

    # deterministic
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.local_features is None:
        print("- Mode: Global module only")
        assert args.global_features is not None
        use_global, use_local = True, False
    elif args.global_features is None:
        print("- Mode: Local module only")
        assert args.local_features is not None
        use_global, use_local = False, True
    else:
        print("- Mode: Full model with global module and local module")
        use_global, use_local = True, True

    print("- Loading training dataset ...")
    dataset = prepare_data.load_multiple_datasets(args.train_dirs, args.local_features, global_features=args.global_features)
    train_data, validate_data = EPI_misc.resplit_train_test(dataset, dataset['chrom'], test_chroms=args.validate_chrs)
    print("- Train data size: {} {} {}".format(
        "enhancer {}".format(train_data['enhancer'].shape) if use_local else "", 
        "promoter {}".format(train_data['promoter'].shape) if use_local else "", 
        "segment {}".format(train_data['segment'].shape) if use_global else ""))
    print("- Validate data size: {} {} {}".format(
        "enhancer {}".format(validate_data['enhancer'].shape) if use_local else "", 
        "promoter {}".format(validate_data['promoter'].shape) if use_local else "", 
        "segment {}".format(validate_data['segment'].shape) if use_global else ""))


    gpu_id = idle_gpu(min_memory=2800)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("- Using GPU: %s" % gpu_id)

    params = json.load(open(args.model_config)) # {"description": xxx, "params": xxx}
    if use_local:
        params['use_local'] = True
        params['e_channel'] = train_data['enhancer'].shape[1]
        params['p_channel'] = train_data['promoter'].shape[1]
    else:
        params['use_local'] = False
    if use_global:
        params['use_global'] = True
        params['s_channel'] = train_data['segment'].shape[1]
    else:
        params['use_global'] = False
    print("- Model config: {}".format(params))

    model_name = args.model_name

    model = EPIGL(**params)
    model.to(device)

    if args.freeze_CNN:
        model.freeze_CNN()
    if args.freeze_LSTM:
        model.freeze_LSTM()
    if args.freeze_ATT:
        model.freeze_ATT()

    if args.opt is not None:
        warnings.warn("--opt -> --optim")
        args.optim = args.opt

    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.model_state is not None:
        print("- Loading model state from {}".format(args.model_state))
        state_dict = torch.load(args.model_state)
        model.load_state_dict(state_dict['model_state_dict'])
        if args.optimizer_state is not None:
            print("- Loading optimizer state from {}".format(args.optimizer_state))
            state_dict = torch.load(args.optimizer_state)
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            
    elif args.optimizer_state is not None:
        warnings.warn("Optimizer state specified but model state not! {} ignored.".format(args.optimizer_state))

    print("- Model\n", model, '\n', biock.model_summary(model))
    print("- Optimizer\n", optimizer)

    print("- Model will be saved to {}/{}".format(outdir, model_name))
    print("- Start training {}".format(time.asctime()))
    model = train_model(model, \
            optimizer,
            train_data, \
            validate_data=validate_data, \
            sample_idxs='all', \
            checkpoint="{}/{}".format(outdir, model_name), \
            batch_size=args.batch_size, \
            epoch=args.epoch, \
            shuffle=True, \
            device=device, \
            att_C=args.att_C, \
            freeze_CNN=args.freeze_CNN,
            freeze_LSTM=args.freeze_LSTM,
            freeze_ATT=args.freeze_ATT,
            learning_curve="{}/{}.learning_curve.pdf".format(outdir, model_name)
        )
    print("- Pretrain finished at {}\n\n".format(time.asctime()))

