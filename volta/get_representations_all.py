# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

from scipy import spatial

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from volta.config import BertConfig
from volta.encoders import BertForVLTasks, BertForVLPreTraining
from volta.task_utils import LoadDatasetEval, LoadLoss, ForwardModelsTrain, ForwardModelsVal

from scipy.stats.mstats import spearmanr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

"""
Read vocabulary bert_base_uncased and store it in list
from: https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
"""
bert_vocab = open('bert_base_uncases_vocab.txt','r')
content = bert_vocab.readlines()
hfvocab = []
for i in content:
    hfvocab.append(str(i.strip()))
print('BERT vocabulary loaded! Len:', len(hfvocab))

"""
Read target vocabulary containing 2278 words from benchmarks
"""
tvocab = open('/project/dmg_data/MMdata/unique-words-filtered-2278.txt','r')
lines = tvocab.readlines()
myvocab = []
for l in lines:
    l = l.strip()
    myvocab.append(str(l.lower()))
print('My vocabulary loaded! Len:', len(myvocab))

bench_path = '/project/dmg_data/MMdata/benchmarks_filtered'

def load_benchmark(jsonfile):
    with open(jsonfile, 'r') as mybench:
        data = json.load(mybench)
        listw1, listw2, listscores = [], [], []
        for k, el in data.items():
            w1 = el['word1'].lower()
            w2 = el['word2'].lower()
            score = el['score']
            listw1.append(w1)
            listw2.append(w2)
            listscores.append(score)
    return listw1, listw2, listscores

def store_repr(matrix, representations):
    for w, l in matrix.items():
        if len(l) == 0:
            avg = []
        elif len(l) == 1:
            avg = torch.cat(l)
        else:
            ll = torch.cat(l)
            lr = torch.reshape(ll,(len(l), 768)) # TODO: avoid hardcoding
            avg = torch.mean(lr, 0)
            if torch.isnan(avg).any() == True:
                print(w, 'has nan', avg)
        representations[w].append(avg)
    return representations


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/bert_config.json", type=str,
                        help="The config file which specified the model details.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Text
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot evaluation.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")

    # Layer
    parser.add_argument("--targ_layer", type=int, default=-1,
                        help="specify the layer from which to get represenations")

    # Modality
    parser.add_argument("--modality", type=str, default="lang",
                        help="specify modality stream you need repr. from: mm, lang, vis")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id

    # Output dirs
    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained
    savePath = os.path.join(args.output_dir, timeStamp)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config, task_cfg, args.task)

    # Model
    if args.zero_shot:
        config.visual_target_weights = {}
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config)
    else:
        model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
        
    # Move to GPU(s)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)
        raise ValueError("Please run with a single GPU")

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters)
        print("  Batch size: ", batch_size)

    """
    - num iters: will be number of sentences in dataset, i.e. around 113K
    - batch size: 1
    """

    # Evaluate
    model.eval()
    
    if args.modality == "lang":
        word_matrices = {w: [] for w in myvocab}
        word_representations = {w: [] for w in myvocab}
    elif args.modality == "vis":
        word_matrices_vv = {w: [] for w in myvocab}
        word_representations_vv = {w: [] for w in myvocab}
    elif args.modality == "mm":
        word_matrices_mm = {w: [] for w in myvocab}
        word_representations_mm = {w: [] for w in myvocab}

    word_occurences = {w: 0 for w in myvocab}

    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):

        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, segment_ids, target, caption_idx, image_idx = batch
 
        myseq, mystrings = [], [] # BERT ids and subswords
        idseq = [] # ids in sentence
        for n, el in enumerate(question[0,:]):
            glolist, glostring = [], []
            loclist = []
            idx = int(el.item()) # e.g. 101, 25, 564, etc.
            if idx != 0:
                tok = hfvocab[idx] # e.g. 'cat', '##ncia', etc.
                if tok.startswith('##'):
                    glolist = myseq[-1]
                    glostring = mystrings[-1]
                    glolist.append(idx)
                    glostring.append(tok)
                    loclist = idseq[-1]
                    loclist.append(n)
                else:
                    glolist.append(idx)
                    myseq.append(glolist) # [[123]] or [[1289], [43], [2352] ]
                    glostring.append(tok) # [[cat]] or [[trans], [##for], [##mer]]
                    mystrings.append(glostring)
                    loclist.append(n)
                    idseq.append(loclist)

        # print(myseq, mystrings)
        # print(idseq, mystrings) # [0], [1], [[2] [3]]

        targlist, targwords = [], []
        for pos, item in enumerate(mystrings):
            if len(item) == 1:
                word = str(item[0])
            else:
                word = ''
                for el in item:
                    word = word+el.replace('#','')
            if word in myvocab:
                targwords.append(word)
                targlist.append(idseq[pos])
                word_occurences[word] += 1
                """
                this word occurrences count can be used at the end
                to check the frequency of each word in the vocabulary
                """
        # print(targlist)
        # print(targwords)

        features = features.squeeze(0)
        spatials = spatials.squeeze(0)
        image_mask = image_mask.squeeze(0)
        question = question.repeat(features.size(0), 1)
        segment_ids = segment_ids.repeat(features.size(0), 1)
        input_mask = input_mask.repeat(features.size(0), 1)

        with torch.no_grad():
            if args.zero_shot:
                _, _, vil_logit, _, _, hid_s_T, hid_s_V = model(question, features, spatials, segment_ids, input_mask,
                                        image_mask, output_all_encoded_layers=True)

                """
                here we select the layer to extract representations from
                e.g. with -1 we get representations from the very last one
                """

                mylayer = int(args.targ_layer)
                # with -1 we consider the last layer:
                # e.g., for ViLBERT, it's the number 36 (len 37)
                # e.g., for LXMERT, it's the number 33 (len 34)

                pL = hid_s_T[mylayer]
                targ_sentence = pL[0] # only 0 works

                pV = hid_s_V[mylayer]
                p_targ_img = pV[0] # only 0 works
                imageX = p_targ_img[0] # the 'cls' of the image
                                
                for idx, w in enumerate(targwords): # e.g. dog, hat, crocheted
                    lmat = []
                    pos = targlist[idx] # position or positions of target word
                    for nn,p in enumerate(pos):
                        lmat.append(targ_sentence[p])
                    if len(lmat) > 1:
                        tempmat = torch.cat(lmat)
                        tm = torch.reshape(tempmat,(len(lmat), 768)) # TODO: remove hardcoding
                        wordX = torch.mean(tm,0)
                    else:
                        wordX = torch.cat(lmat)

                    """
                    save representations!
                    """
                    if args.modality == "lang":
                        if torch.isnan(wordX).any() == True:
                            continue
                        else:
                            word_matrices[w].append(wordX)
                    elif args.modality == "vis":
                        word_matrices_vv[w].append(imageX)
                    elif args.modality == "mm":
                        mmX = imageX * wordX
                        word_matrices_mm[w].append(mmX)

                del hid_s_T
                del hid_s_V
                del pL 
                del pV
                del vil_logit
                torch.cuda.empty_cache()

            else:
                print('To be implemented')
                exit(0)

    torch.cuda.empty_cache()

    if args.modality == "lang":
        word_representations = store_repr(word_matrices, word_representations)
    elif args.modality == "vis":
        word_representations_vv = store_repr(word_matrices_vv, word_representations_vv)
    elif args.modality == "mm":
        word_representations_mm = store_repr(word_matrices_mm, word_representations_mm)

    outall = open(args.output_dir+'/correlations_layers_'+args.modality+'.txt', 'a')
    for filename in os.listdir(bench_path):
        if filename.endswith('.json'):
            targname = filename.split('.')[0]
            out = open(args.output_dir+'/'+targname+'_'+args.modality+'_'+
                    str(args.targ_layer)+'.txt', 'w')
            out.write('idx,modality,layer,word1,word2,hum-sim,model-sim,freqw1,freqw2'+'\n')
            w1s, w2s, GTscores = load_benchmark(bench_path+'/'+filename)
            print('Computing similarities for:',filename)
            Lscores, Vscores, MMscores = [],[],[]
            foundscores = []
            for npair in range(len(GTscores)):
                c = 0
                w1 = w1s[npair]
                if args.modality == "lang":
                    l1out = word_representations[w1][0]
                    if len(l1out) > 0:
                        rl1out = l1out.cpu().numpy()
                        c+=1
                elif args.modality == "mm":
                    mm1out = word_representations_mm[w1][0]
                    if len(mm1out) > 0:
                        rmm1out = mm1out.cpu().numpy()
                        c+=1
                elif args.modality == "vis":
                    vv1out = word_representations_vv[w1][0]
                    if len(vv1out) > 0:
                        rvv1out = vv1out.cpu().numpy()
                        c+=1
                w2 = w2s[npair]
                if args.modality == "lang":
                    l2out = word_representations[w2][0]
                    if len(l2out) > 0:
                        rl2out = l2out.cpu().numpy()
                        c+=1
                elif args.modality == "mm":
                    mm2out = word_representations_mm[w2][0]
                    if len(mm2out) > 0:
                        rmm2out = mm2out.cpu().numpy()
                        c+=1
                elif args.modality == "vis": 
                    vv2out = word_representations_vv[w2][0]
                    if len(vv2out) > 0:
                        rvv2out = vv2out.cpu().numpy()
                        c+=1
                if c==2:
                    if args.modality == "lang":
                        simL = 1 - spatial.distance.cosine(rl1out,rl2out)
                        Lscores.append(simL)
                        out.write(str(npair)+','+args.modality+','+str(args.targ_layer)+','+w1+','+w2+','+
                                str(GTscores[npair])+','+str(simL)+','+
                                str(len(word_matrices[w1]))+','+str(len(word_matrices[w2]))+'\n')
                    elif args.modality == "mm":
                        simMM = 1 - spatial.distance.cosine(rmm1out,rmm2out)
                        MMscores.append(simMM)
                        out.write(str(npair)+','+args.modality+','+str(args.targ_layer)+','+w1+','+w2+','+
                                str(GTscores[npair])+','+str(simMM)+','+
                                str(len(word_matrices_mm[w1]))+','+str(len(word_matrices_mm[w2]))+'\n')
                    elif args.modality == "vis":
                        simV = 1 -spatial.distance.cosine(rvv1out, rvv2out)
                        Vscores.append(simV)
                        out.write(str(npair)+','+args.modality+','+str(args.targ_layer)+','+w1+','+w2+','+
                                str(GTscores[npair])+','+str(simV)+','+
                                str(len(word_matrices_vv[w1]))+','+str(len(word_matrices_vv[w2]))+'\n')
                    foundscores.append(GTscores[npair])

        print('Representation from stream:', args.modality)
        print('Correlation for the following layer:', args.targ_layer)
        if args.modality == "lang":
            if len(foundscores) > 2:
                Lcorr = spearmanr(foundscores, Lscores)
                print('Pairs found:', len(Lscores), len(foundscores), len(GTscores))
                print('L correlation:', Lcorr)
                outall.write(targname+','+str(args.targ_layer)+','+str(len(Lscores))+','+
                    str(Lcorr[0])+','+str(Lcorr[1])+'\n')
        elif args.modality == "vis":
            Vcorr = spearmanr(foundscores, Vscores)
            print('Pairs found:', len(Vscores), len(foundscores), len(GTscores))
            print('V correlation:', Vcorr)
        elif args.modality == "mm":
            MMcorr = spearmanr(foundscores, MMscores)
            print('Pairs found:', len(MMscores), len(foundscores), len(GTscores))
            print('MM correlation:', MMcorr)
        out.close()

if __name__ == "__main__":
    main()
