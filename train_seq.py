from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_seq.crf import CRFLoss, CRFDecode
from model_seq.dataset import SeqDataset
from model_seq.evaluator import eval_wc
from model_seq.seqlabel import Vanilla_SeqLabel
import model_seq.utils as utils

from torch_scope import wrapper

import argparse
import json
import os
import sys
import itertools
import functools


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='./checkpoint')
    parser.add_argument('--checkpoint_name', default='ner')
    parser.add_argument('--git_tracking', action='store_true')

    parser.add_argument('--corpus', default='./data/ner_dataset.pk')

    parser.add_argument('--seq_c_dim', type=int, default=30)
    parser.add_argument('--seq_c_hid', type=int, default=150)
    parser.add_argument('--seq_c_layer', type=int, default=1)
    parser.add_argument('--seq_w_dim', type=int, default=100)
    parser.add_argument('--seq_w_hid', type=int, default=300)
    parser.add_argument('--seq_w_layer', type=int, default=1)
    parser.add_argument('--seq_droprate', type=float, default=0.5)
    parser.add_argument('--seq_model', choices=['vanilla'], default='vanilla')
    parser.add_argument('--seq_rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='SGD')
    args = parser.parse_args()

    # automatically sync to spreadsheet
    # pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, \
    #                   sheet_track_name=args.spreadsheet_name, credential_path="/data/work/jingbo/ll2/Torch-Scope/torch-scope-8acf12bee10f.json")
    
    pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking)
    pw.set_level('info')

    gpu_index = pw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
    
    pw.info('Loading data')

    dataset = pickle.load(open(args.corpus, 'rb'))
    name_list = ['gw_map', 'c_map', 'y_map', 'emb_array', 'train_data', 'test_data', 'dev_data']
    gw_map, c_map, y_map, emb_array, train_data, test_data, dev_data = [dataset[tup] for tup in name_list ]

    
    pw.info('Building models')

    SL_map = {'vanilla':Vanilla_SeqLabel}
    seq_model = SL_map[args.seq_model](len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map), args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate, unit=args.seq_rnn_unit)
    seq_model.rand_init()
    seq_model.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    seq_config = seq_model.to_params()
    seq_model.to(device)
    crit = CRFLoss(y_map)
    decoder = CRFDecode(y_map)
    evaluator = eval_wc(decoder, 'f1')

    pw.info('Constructing dataset')

    train_dataset, test_dataset, dev_dataset = [SeqDataset(tup_data, gw_map['<\n>'], c_map[' '], c_map['\n'], y_map['<s>'], y_map['<eof>'], len(y_map), args.batch_size) for tup_data in [train_data, test_data, dev_data]]

    pw.info('Constructing optimizer')

    param_dict = filter(lambda t: t.requires_grad, seq_model.parameters())
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](param_dict, lr=args.lr)
    else:
        optimizer=optim_map[args.update](param_dict)

    pw.info('Saving configues.')
    pw.save_configue(args)

    pw.info('Setting up training environ.')
    best_f1 = float('-inf')
    patience_count = 0
    batch_index = 0
    normalizer=0
    tot_loss = 0

    try:
        for indexs in range(args.epoch):

            pw.info('############')
            pw.info('Epoch: {}'.format(indexs))
            pw.nvidia_memory_map()

            seq_model.train()
            for f_c, f_p, b_c, b_p, f_w, f_y, f_y_m, _ in train_dataset.get_tqdm(device):

                seq_model.zero_grad()
                output = seq_model(f_c, f_p, b_c, b_p, f_w)
                loss = crit(output, f_y, f_y_m)

                tot_loss += utils.to_scalar(loss)
                normalizer += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(seq_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1
                if 0 == batch_index % 100:
                    pw.add_loss_vs_batch({'training_loss': tot_loss / (normalizer + 1e-9)}, batch_index, use_logger = False)
                    tot_loss = 0
                    normalizer = 0

            if args.lr > 0:
                current_lr = args.lr / (1 + (indexs + 1) * args.lr_decay)
                utils.adjust_learning_rate(optimizer, current_lr)

            dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))

            pw.add_loss_vs_batch({'dev_f1': dev_f1}, indexs, use_logger = True)
            pw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, indexs, use_logger = False)
            
            pw.info('Saving model...')
            pw.save_checkpoint(model = seq_model, 
                        is_best = (dev_f1 > best_f1), 
                        s_dict = {'config': seq_config, 
                                'gw_map': gw_map, 
                                'c_map': c_map, 
                                'y_map': y_map})

            if dev_f1 > best_f1:
                test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))
                best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc
                pw.add_loss_vs_batch({'test_f1': test_f1}, indexs, use_logger = True)
                pw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, indexs, use_logger = False)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    break

    except Exception as e_ins:

        pw.info('Exiting from training early')

        print(type(e_ins))
        print(e_ins.args)
        print(e_ins)

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(seq_model, dev_dataset.get_tqdm(device))

        pw.add_loss_vs_batch({'dev_f1': dev_f1}, indexs, use_logger = True)
        pw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, indexs, use_logger = False)
        
        pw.info('Saving model...')
        pw.save_checkpoint(model = seq_model, 
                    is_best = (dev_f1 > best_f1), 
                    s_dict = {'config': seq_config, 
                            'gw_map': gw_map, 
                            'c_map': c_map, 
                            'y_map': y_map})

        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(seq_model, test_dataset.get_tqdm(device))
        best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc
        pw.add_loss_vs_batch({'test_f1': test_f1}, indexs, use_logger = True)
        pw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, indexs, use_logger = False)

    pw.close()
