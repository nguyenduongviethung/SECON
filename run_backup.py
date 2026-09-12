# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np

from model import Model,CoModel

from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

import torch.distributed as dist
import contextlib
import torch.nn as nn
import math
from copy import deepcopy

from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
import torch.nn.init as init
from torch.nn import Parameter
from torch.optim import Optimizer
import copy

from typing import Optional
from collections import OrderedDict
from torch.nn.modules.module import Module
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import statistics
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 

logger = logging.getLogger(__name__)


def get_expert_split(lang_type, tokenizer, args):
    train_dataset = get_datasets_dic(tokenizer, lang_type, args)
    return train_dataset

def get_datasets_dic(tokenizer, lang_type, args):
    # dataset_dic = {}
    # python 251820 java 164923 go 167288 ruby 24927 javascript 58025 php 241241
    train_dataset = TextDataset(tokenizer, args, os.path.join(args.root_data_file, lang_type, args.train_data_file))
    # dataset_dic[lang_type] = train_dataset
    return train_dataset
def covariance_loss(z1: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D 
    
    return cov_loss

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def polyloss(view1, view2, margin):
    
    sim_mat = sim_matrix(view1,view2)
    epsilon = 1e-5
    size=sim_mat.size(0)
    hh=sim_mat.t()
    label=torch.Tensor([i for i in range(size)])
  
    loss = list()
    for i in range(size):
        pos_pair_ = sim_mat[i][i]
        pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        neg_pair_ = sim_mat[i][label!=label[i]]

        neg_pair = neg_pair_[neg_pair_ + margin > min(pos_pair_)]

        pos_pair=pos_pair_
        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue

        pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5, min=0)
        neg_pair=max(neg_pair)
        neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)

        loss.append(pos_loss + neg_loss)
    for i in range(size):
        pos_pair_ = hh[i][i]
        pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        neg_pair_ = hh[i][label!=label[i]]

        neg_pair = neg_pair_[neg_pair_ + margin > min(pos_pair_)]

        pos_pair=pos_pair_
        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue
        pos_loss =torch.clamp(0.2*torch.pow(pos_pair,2)-0.7*pos_pair+0.5,min=0)

        neg_pair=max(neg_pair)
        neg_loss = torch.clamp(0.9*torch.pow(neg_pair,2)-0.4*neg_pair+0.03,min=0)
        loss.append(pos_loss + neg_loss)
        
    if len(loss) == 0:
        return torch.zeros([], requires_grad=True)

    loss = sum(loss) / size
    return loss

    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    
    return sim_mt  


def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    # 代码片段预处理
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    # 将token 转换成 id
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    # 加padding 补齐
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    # 自然语言预处理
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                # 对每行进行处理
                for line in f:
                    # 删除空格和换行符和t
                    line = line.strip()
                    # json.load()方法是从json文件读取json，而json.loads()方法是直接读取json，两者都是将字符串json转换为字典
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for js in data:
            # 将js字典转换
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
            '''
            examples包括以下几个部分
                    self.code_tokens = code_tokens
                    self.code_ids = code_ids
                    self.nl_tokens = nl_tokens
                    self.nl_ids = nl_ids
                    self.url = url
            '''
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
    
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
  
class CustomDataset(Dataset):
    def __init__(self, code, nl):
        self.code = code
        self.nl = nl

    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        return self.code[idx], self.nl[idx]

def save_model(model, name, epoch, test_way='ood'):
    if not os.path.exists("model/code_nl"):
        os.makedirs("model/code_nl")
    path = "model/code_nl/{}_{}_best.bin".format(name, epoch)
    torch.save(model.state_dict(), path)


def loss_fn(nl_vec,code_vec):

    poly_loss = polyloss(nl_vec,code_vec,0.15)

    return poly_loss

class NormSoftmaxLoss(nn.Module):
    #https://github.com/TencentARC/MCQ/blob/3555a9bbebca0919eebfe1c4b398c3686057ef77/MILES/model/loss.py
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, view1, view2):
        x = sim_matrix(view1, view2)

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        #print(x.shape)
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
    
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_names=["word_embeddings"]):
        # emb_name这个参数要换成你模型中embedding的参数名
        for emb_name in emb_names:
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_names=["word_embeddings"]):
        # emb_name这个参数要换成你模型中embedding的参数名#
        for emb_name in emb_names:
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name: 
                    assert name in self.backup
                    param.data = self.backup[name]
            self.backup = {}
                        
            
def compute_gce(z1_outs, z2_outs, quantile):
      # (1) Stack and normalize outputs
      src_train_full = torch.nn.functional.normalize(torch.vstack(z1_outs)).cuda()
      tgt_train_full = torch.nn.functional.normalize(torch.vstack(z2_outs)).cuda()
      z1_outs, z2_outs = [], []

      # (2) Estimate quantile
      chunk_size, num_samples, quantiles = 10, len(tgt_train_full), []
      for chunk_idx in range(math.ceil(len(tgt_train_full)/chunk_size)):
        quantiles.append(float(torch.quantile((src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T), quantile)))

      # (3) Get similarity graph thresholded on quantile
      row, col, data, quantile = [], [], [], statistics.median(quantiles)
      #print(quantile)
      for chunk_idx in range(math.ceil(len(src_train_full)/chunk_size)):
        ret = ((src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T).flatten() > quantile).nonzero(as_tuple=True)[0].cpu()
        row += ((ret - (ret % num_samples))/num_samples + chunk_idx*chunk_size).int().tolist()
        col += (ret % num_samples).tolist()
        data += [1.0 for _ in range(len(ret))]

      # (4) Get permutation using graph bandwidth minimization on sparsified graph (cuthill-mckee)
      #print(len(data)/(len(src_train_full)**2))
      permutation = list(reverse_cuthill_mckee(csr_matrix((data, (row, col)), 
                                                    shape=(num_samples, num_samples))))
      return permutation
    
    
def train(args, model, cmodel, tokenizer):
        
    train_data = {}
    for lang_type in ["ruby", "javascript", "go", "java", "php", "python"]:
        try:
            with open(f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl", "rb") as f:
                train_data[lang_type] = pickle.load(f)
        except FileNotFoundError:
        # 字典存储属于某类 data: class
            train_data[lang_type] = get_expert_split(lang_type, tokenizer, args)
            file_path = f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(f"model/{lang_type}/pre_embedding_num_{args.datasets_len}_batchSize_{args.train_batch_size}.pkl", "wb") as f:
                pickle.dump(train_data[lang_type], f)
    
    """ Train the model """
    for lang_type in ["ruby", "javascript", "go", "java", "php", "python"]:
        #get training dataset
        train_sampler = RandomSampler(train_data[lang_type])
        train_dataloader = DataLoader(train_data[lang_type], sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
        '''
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
        '''
        #get optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

        coptimizer = AdamW(cmodel.parameters(), lr=args.learning_rate, eps=1e-8)
        cscheduler = get_linear_schedule_with_warmup(coptimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data[lang_type]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
        logger.info("  Total train batch size  = %d", args.train_batch_size)
        logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
        
        # model.resize_token_embeddings(len(tokenizer))
        model.zero_grad()

        cmodel.zero_grad()
            
        cmodel.train()      

        model.train()        

        #print(model.module.gpool)
        #print(model.module.encoder.encoder.layer[11])
        tr_num,tr_loss,best_mrr = 0,0,0 

        for idx in range(args.num_train_epochs): 
            '''
            model.eval()
            with torch.no_grad():
                eval_z1, eval_z2 = [], []
                for batch in train_dataloader:
                    #get code and nl vectors
                    code_inputs = batch[0].to(args.device)    
                    nl_inputs = batch[1].to(args.device)
                    code_vec= model(code_inputs=code_inputs)
                    nl_vec= model(nl_inputs=nl_inputs)
                    
                    eval_z1.append(code_vec.cpu().float())
                    eval_z2.append(nl_vec.cpu().float())

                permutation = compute_gce(eval_z1, eval_z2, 0.999)

                train_dataset = torch.utils.data.Subset(train_dataset, permutation)
                train_sampler = SequentialSampler(train_dataset)
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            model.train()
            '''
            for step,batch in enumerate(train_dataloader):
                #get inputs
                code_inputs = batch[0].to(args.device)    
                nl_inputs = batch[1].to(args.device)
    
                #get code and nl vectors
                
                code_vec = model(code_inputs=code_inputs)
                nl_vec = model(nl_inputs=nl_inputs)

                vec1,vec2 = cmodel(code_inputs=code_inputs,nl_inputs=nl_inputs)

                loss =  loss_fn(code_vec,nl_vec) + loss_fn(nl_vec,vec1) + loss_fn(vec2,code_vec) + covariance_loss(code_vec)

                
                #report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step+1)%100 == 0:
                    logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                    tr_loss = 0
                    tr_num = 0
                
                #backward
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() 

                torch.nn.utils.clip_grad_norm_(cmodel.parameters(), args.max_grad_norm)
                coptimizer.step()
                coptimizer.zero_grad()
                cscheduler.step() 



            #evaluate    
            #results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
            results = evaluate(args, model, tokenizer,lang_type, args.eval_data_file, eval_when_training=True)

            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4)) 

            #save best model
            if results['eval_mrr']>best_mrr:
                best_mrr = results['eval_mrr']
                logger.info("  "+"*"*20)  
                logger.info("  Best mrr:%s",round(best_mrr,4))
                logger.info("  "+"*"*20)  


                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)



def evaluate(args, model, tokenizer,lang_type, file_sub_name, eval_when_training=False):
    file_name = os.path.join(args.root_data_file, lang_type, file_sub_name)
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    file_name_codebase = os.path.join(args.root_data_file, lang_type, args.codebase_file)
    code_dataset = TextDataset(tokenizer, args, file_name_codebase)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }
        

    return result

                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--root_data_file", default="dataset/CSN", type=str,
                    help="An optional input test data file to test the MRR(a josnl file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--datasets_len", default="12345678910", type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
        
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)#(args.model_name_or_path)
    #config = RobertaConfig.from_pretrained(args.model_name_or_path)  ("DeepSoftwareAnalytics/CoCoSoDa")#
    model = RobertaModel.from_pretrained(args.model_name_or_path)#(args.model_name_or_path) 
    model2 = RobertaModel.from_pretrained(args.model_name_or_path)#(args.model_name_or_path) 

    model = Model(model)
    cmodel = CoModel(model2, args)

    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    cmodel.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
        cmodel = torch.nn.DataParallel(cmodel)  

    # Training
    if args.do_train:
        train(args, model, cmodel, tokenizer)
    '''
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
    '''

if __name__ == "__main__":
    main()