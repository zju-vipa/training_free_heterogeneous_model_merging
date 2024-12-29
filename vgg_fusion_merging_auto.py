import os

import argparse
parser = argparse.ArgumentParser('Evaluation mudsc merge')
from thop import profile
import math
from ptflops import get_model_complexity_info
parser.add_argument('--config-name', type=str,
                        help='config name')


parser.add_argument('--config-name-b', type=str,
                        help='config name')


parser.add_argument('--suffix', default="", type=str,
                        help='config name')

parser.add_argument('--gpu', default='0',type=str,
                        help='gpu')

args = parser.parse_args()

print("config name", args.config_name,"suffix",args.suffix,"setting gpu",args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch import nn
import random
from time import time
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import gc
from utils import inject_pair,prepare_experiment_config,reset_bn_stats,write_to_csv
from utils import evaluate_model,train_model,CONCEPT_TASKS,flatten_nested_dict,get_config_from_name,find_runable_pairs
from graphs.vgg_graph import VGGGraph
from models.vgg import cfg as vgg_cfg
from model_merger import ModelMerge
from matching_functions import match_tensors_zipit,match_tensors_permute
from metric_calculators import CovarianceMetric
import torch.backends.cudnn as cudnn
import pickle as pkl
from typing import List
import re


suffix = args.suffix


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import pandas as pd


def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, list):
            if k not in d:
                d[k] = []
            d[k] += v
        else:
            d[k] = v
    return d


def create_df(search_results):
    base = {}
    for _, results in search_results.items():
        base = dict_update(base, results)

    numbers = np.array(list(base.values())).T
    cols = list(base.keys())

    df = pd.DataFrame(numbers, columns=cols)
    return df


def get_task_mapping(labels, splits):
    task_mapping = []
    for i, label in enumerate(labels):
        for j, split in enumerate(splits):
            if label in split:
                task_mapping.append(j)
    return torch.from_numpy(np.array(task_mapping))

def get_identity_idx(n,m, weight):
    s = min(m,n)
    s = torch.arange(s,device=weight.device)
    return s, s 

@torch.no_grad()
def dummify_model(model:nn.Module):
    for na,m in model.named_modules():
        if len(list(m.parameters(recurse=False)))==0:
            continue
        if isinstance(m,nn.Conv2d):
            m.weight.fill_(0)
            mid_ = m.weight.shape[2]//2
            
            row, col = get_identity_idx(*m.weight.shape[:2],m.weight)
            # val = 1 if "shortcut" in na else 0
            val = 1
            m.weight.data[row, col,mid_,mid_] = val
            if m.bias is not None:
                m.bias.fill_(0)
        elif isinstance(m,nn.Linear):
            m.weight.fill_(0)
            row, col = get_identity_idx(*m.weight.shape[:2],m.weight)
            m.weight.data[row, col] = 1
            if m.bias is not None:
                m.bias.fill_(0)
        elif isinstance(m,nn.BatchNorm2d):
            # val = 1 if "shortcut" in na else 0
            val = 1
            m.weight.fill_(val)
            m.bias.fill_(0)
            m.running_mean.fill_(0)
            m.running_var.fill_(val)
        else:
            print(f"Not Implement class {m.__class__.__name__}")
            raise NotImplementedError
    
# def trim_units(model:nn.Module,named_modules):
#     for na,m in model.named_modules():
#         if na.split(".")[-1] != "shortcut" or na in named_modules:
#             continue
#         bn = m[0].
from torch_cka import CKA
def align(C):
    """
    Compute the optimal map between hidden layers of two models using dynamic programming.
    
    :param C: cost matrix
    :param start_layer_idx: the layer index to start aligning, default = 2, 
        i.e., start from the second hidden layer
    :param free_end_layers: match last free_end_layers hidden layers 
        of two models, default = 0
    :return: list of layer indices of the large model
    """
    m, n = C.shape
    assert m >= n

    F = torch.zeros((n+1, m+1))

    # compute the diagonal of F
    sum = 0
    for k in range(1, n+1):
        sum += C[k-1, k-1]
        F[k, k] = sum

    # forward recursion
    for k in range(1, n+1):
        for l in range(k+1, m+1):
            F[k, l] = max(F[k, l-1], F[k-1, l-1] + C[l-1, k-1])
    
    # backward recursion 
    A = torch.ones(n+1)
    k, l = n, m
    print(F)
    while (k >= 0):
        while (l >= k+1) and (F[k, l] == F[k, l-1]):
            l -= 1

        A[k] = l
        k -= 1
        l -= 1

    # because the first hidden layer is layer at index 1
    return A[1:]-1

def align2(C):
    """
    Compute the optimal map between hidden layers of two models using dynamic programming.
    
    :param C: cost matrix
    :param start_layer_idx: the layer index to start aligning, default = 2, 
        i.e., start from the second hidden layer
    :param free_end_layers: match last free_end_layers hidden layers 
        of two models, default = 0
    :return: list of layer indices of the large model
    """
    m, n = C.shape
    assert m >= n

    F = torch.zeros((n+1, m+1))

    # compute the diagonal of F
    sum = 0
    for k in range(1, n+1):
        sum += C[k-1, k-1]
        F[k, k] = sum

    # forward recursion
    for k in range(1, n+1):
        for l in range(k+1, m+1):
            F[k, l] = max(F[k, l-1] + C[l-1, k-2], F[k-1, l-1] + C[l-1, k-1])
    
    
    # backward recursion 
    A = torch.ones(n+1)
    k, l = n, m

    while (k >= 0):
        while (l >= k+1) and (F[k, l] == F[k, l-1] + C[l-1, k-2]):
            l -= 1

        A[k] = l
        k -= 1
        l -= 1

    #because the first hidden layer is layer at index 1
    return A[1:]-1

def get_matrix(model_a, model_b, dataloader):
    """
    Compute the matrix of pairwise CKA between layers of a model.
    
    :param model: a model
    :param dataloader: a dataloader
    :return: matrix of pairwise CKA
    """
    cka = CKA(model_a, model_b, device = 'cuda', modeltype = 'ResNet')
    matrix = cka.compare(dataloader)
    return matrix

def sma_align(model_a,model_b,train_loader,stages_a,stages_b):
    """
    model_a: 
    model_b:
    train_loader:
    stages_a,stages_b: 模型的不同阶段，输入格式如: [5,5,5],表示模型有3个stages,每个stage包含5个block/layer
    """
    sim_mat = get_matrix(model_a, model_b, train_loader) # 根据model_a,model_b,train_loader得到的层相似性矩阵

    result = []
    for num_layer_a,num_layer_b in zip(stages_a,stages_b):
        sta,stb = 0,0
        A = align(sim_mat[sta+1:sta+num_layer_a,stb+1:stb+num_layer_b])
        result.append({str(i+1):str(int(A[i].detach().item())+1) for i in range(num_layer_b-1)})
        sta += num_layer_a
        stb += num_layer_b
    return result

def lma_align(model_a,model_b,train_loader,stages_a,stages_b):
    """
    model_a: 
    model_b:
    train_loader:
    stages_a,stages_b: 模型的不同阶段，输入格式如: [5,5,5],表示模型有3个stages,每个stage包含5个block/layer
    """
    sim_mat = get_matrix(model_a, model_b, train_loader) # 根据model_a,model_b,train_loader得到的层相似性矩阵

    result = []
    for num_layer_a,num_layer_b in zip(stages_a,stages_b):
        sta,stb = 0,0
        A = align2(sim_mat[sta+1:sta+num_layer_a,stb+1:stb+num_layer_b])
        result.append({str(i+1):str(int(A[i].detach().item())+1) for i in range(num_layer_b-1)})
        sta += num_layer_a
        stb += num_layer_b
    return result


def generate_weight_remap(blocks1,blocks2, state_dict2 ,verbose = True,model_a=None,model_b=None,act_loader=None):
    weight_remapping = {}
    layer_map = []
    if "headalign" in suffix:
        return {}
    elif "unialign" in suffix:
        # uniform alignment
        for l1,l2 in zip(blocks1,blocks2):
            skip = int(l1/l2 + 0.5)
            layer_map.append({str(k):str(v) for k,v in zip(range(l2-1,0,-1),range(l1-1,0,-skip))})
    elif "htalign" in suffix:
    # head tail
        for l1,l2 in zip(blocks1,blocks2):
            half = int(l2/2)
            layer_map.append({str(k):str(v) for k,v in zip(range(l2-1,half-1,-1),range(l1-1,0,-1))})
    elif "smaalign" in suffix:
        layer_map = sma_align(model_a,model_b,act_loader,blocks1,blocks2)
    elif "lmaalign" in suffix:
        layer_map = lma_align(model_a,model_b,act_loader,blocks1,blocks2)
    else:
        return None
    if verbose:
        print("Layer Mapping", layer_map)
    for k,_ in state_dict2.items():
        if k[:len("layer")] != "layer":
            continue
        ks = k.split(".")
        layer = int(ks[0][len("layer"):])
        block = ks[1]
        if block in layer_map[layer - 1]:
            ks[1] = layer_map[layer - 1][block]
        newk = ".".join(ks)
        assert newk not in weight_remapping
        weight_remapping[k] = newk
        
    if verbose:
        print("Weight Remapping", weight_remapping)
    return weight_remapping

def align_models(base_models_a:List[nn.Module],base_models_b:List[nn.Module],weight_remap = {}):
    if len(base_models_a[0].state_dict()) > len(base_models_b[0].state_dict()):
        base_model = deepcopy(base_models_a[0])
        align_models = base_models_b
    else:
        base_model = deepcopy(base_models_b[0])
        align_models = base_models_a
    dummify_model(base_model)

    raw_models:List[nn.Module] = []
    raw_models.extend(align_models)
    align_models.clear()
    
    for m in raw_models:
        bm = deepcopy(base_model)
        state_dict = {weight_remap[k] if k in weight_remap else k:v for k,v in m.state_dict().items()}
        bm.load_state_dict(state_dict,strict=False)
        # trim_units(bm, [na for na,_ in m.named_modules()])
        align_models.append(bm)

class SharedBackboneModel(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""

    def __init__(self, backbone, heads):
        super(SharedBackboneModel, self).__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList(heads)
        self.train_backbone = True

    def train(self, mode: bool = True) -> nn.Module:
        if mode:
            if self.train_backbone:
                self.backbone.train()
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                self.backbone.eval()
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self.heads.train()
        else:
            self.backbone.eval()
            self.heads.eval()
        return self

    def forward(self, x):
        """Call all models returning list of their outputs."""
        x = self.backbone(x)
        x = [h(x) for h in self.heads]
        if len(x) == 1:
            return x[0]
        else:
            return x
class SharedModel(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        """Call all models returning list of their outputs."""
        return [h(x) for h in self.models]

class SharedEnsembleModel(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        """Call all models returning list of their outputs."""
        xs = [h(x) for h in self.models]
        x = xs[0]
        for i in range(1,len(xs)):
            x += xs[i]
        return x / len(xs)

def run_node_experiment(node_config, experiment_config,experiment_config_b, pairs, csv_file):
    all_results = []
    for i, pair in enumerate(tqdm(pairs, desc='Evaluating Pairs...')):
        gc.collect()
        torch.cuda.empty_cache()
        experiment_config = inject_pair(experiment_config, pair)
        experiment_config_b = inject_pair(experiment_config_b, pair)
        config = prepare_experiment_config(experiment_config)
        config_b = prepare_experiment_config(experiment_config_b)
        train_loader = config['data']['train']['full']
        reset_random()
        print("preparing models a")
        base_models_a = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config['models']['bases']
        ]
        print("preparing models b")
        base_models_b = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config_b['models']['bases']
        ]
        # print("preparing models a")
        # base_models_a = [base_model
        #     for base_model in config['models']['bases']
        # ]
        # print("preparing models b")
        # base_models_b = [base_model
        #     for base_model in config_b['models']['bases']
        # ]
        device = list(base_models_a[0].state_dict().values())[0].device
        with torch.cuda.device(device):
            macs, params = get_model_complexity_info(base_models_a[0],(3,32,32),print_per_layer_stat=False)
            print('{}  {:<8}'.format('Model A Computational complexity: ', macs))
            macs, params = get_model_complexity_info(base_models_b[0],(3,32,32),print_per_layer_stat=False)
            print('{}  {:<8}'.format('Model B Computational complexity: ', macs))
        
        print("size", len(base_models_a), len(base_models_b))
        config['node'] = node_config
        config_b['node'] = node_config
        exclude_param_names_regex = [".*linear.*"]
        head_name = "linear"
        
        def get_num_blocks(cfg_name):
            if "vgg19" in cfg_name:
                return [6,6,6]
            elif "vgg13" in cfg_name:
                return [4,4,4]
            else:
                raise NotImplementedError
        def get_arch(cfg_name):
            if "vgg19" in cfg_name:
                return vgg_cfg["VGG19"]
            elif "vgg13" in cfg_name:
                return vgg_cfg["VGG13"]
            else:
                raise NotImplementedError
        architecture=get_arch(config_name)
        def get_activate_loader():
                return config['data']['train']["sample"] if "sample" in config[
                    'data']['train'] else train_loader
        # weight_remap = {}
        for models_name,base_models,pair_idxs in zip([[config['model']['name'], config_b['model']['name']],
                                            [config_b['model']['name'], config['model']['name']]],
                                           [[base_models_a[0],base_models_b[1]],
                                            [base_models_b[0],base_models_a[1]]],[[0,1],[1,0]]):
            raw_models = deepcopy(base_models)
            ba,bb = [deepcopy(base_models[0])], [deepcopy(base_models[1])]
            if not (config_name == config_name_b):
                ma,mb = deepcopy(ba[0]).cuda(),deepcopy(bb[0]).cuda()
       
                if len(ma.state_dict()) < len(mb.state_dict()):
                    ma,mb = mb,ma
                weight_remap = generate_weight_remap(get_num_blocks(config_name),
                                             get_num_blocks(config_name_b),
                                             mb.state_dict(),model_a=ma,model_b=mb,act_loader=get_activate_loader())
                align_models(ba,bb,weight_remap)
            base_models[0], base_models[1] = ba[0],bb[0]
            temp_model = deepcopy(base_models[pair_idxs[0]])
            graphs = [VGGGraph(b,architecture).graphify() for b in base_models]
            merge_model = ModelMerge(*graphs,device=device)

            
            print(f"Mergeing model {models_name[0]} with {models_name[1]}")
            
            reset_random()

            
            gc.collect()
            torch.cuda.empty_cache()
            transform_fn = match_tensors_zipit
            if "perm" in suffix:
                transform_fn = match_tensors_permute
            stop_at = None
            if "102" in suffix:
                stop_at = 102
            merge_model.transform(
                temp_model, 
                get_activate_loader(), 
                transform_fn=transform_fn, 
                metric_classes=(CovarianceMetric,),
                stop_at=stop_at, #153, 102
                **node_config['params']
            )
                
            with torch.cuda.device(device):
                macs, params = get_model_complexity_info(merge_model,(3,32,32),print_per_layer_stat=False)
            print('{}  {:<8}'.format('Merged Model Computational complexity: ', macs))
            # merge_model = base_models[1]
            reset_random()
            reset_bn_stats(merge_model, train_loader)
            gc.collect()
            torch.cuda.empty_cache()
            merge_model.train_backbone = False
            results = evaluate_model(experiment_config['eval_type'],
                                    merge_model,
                                    config,
                                    test_train="train" in suffix,lr=0.4,epochs=10)
            merge_model.eval()
            if pair_idxs[0] != 0:
                results["Task A"], results["Task B"] = results["Task B"], results["Task A"]
            for idx,pair_idx in enumerate(pair_idxs):
                results[f'Split {CONCEPT_TASKS[idx]}'] = pair[pair_idx]
            # results['Time'] = Merge.compute_transform_time
            results['Merging Fn'] = f"merged"
            results['Model A'] = config['model']['name']
            results['Model B'] = config_b['model']['name']
            results.update(flatten_nested_dict(node_config, sep=' '))
            write_to_csv(results, csv_file=csv_file)
            print(results)
            all_results.append(results)
                


            # ensemble
            if False:
                ensemble_model = SharedModel(raw_models)
                with torch.cuda.device(device):
                    macs, params = get_model_complexity_info(ensemble_model,(3,32,32),print_per_layer_stat=False)
                print('{}  {:<8}'.format('Ensemble Model Computational complexity: ', macs))
                results = evaluate_model(experiment_config['eval_type'],
                                        ensemble_model,
                                        config,
                                        test_train="train" in suffix)
                if pair_idxs[0] != 0:
                    results["Task A"], results["Task B"] = results["Task B"], results["Task A"]
                for idx,pair_idx in enumerate(pair_idxs):
                    results[f'Split {CONCEPT_TASKS[idx]}'] = pair[pair_idx]
                # results['Time'] = Merge.compute_transform_time
                results['Merging Fn'] = "two models"
                results['Model A'] = config['model']['name']
                results['Model B'] = config_b['model']['name']
                results.update(flatten_nested_dict(node_config, sep=' '))
                write_to_csv(results, csv_file=csv_file)
                print(results)
            if False:
                gc.collect()
                torch.cuda.empty_cache()
                ensemble_model = SharedEnsembleModel(raw_models)
                with torch.cuda.device(device):
                    macs, params = get_model_complexity_info(ensemble_model,(3,32,32),print_per_layer_stat=False)
                print('{}  {:<8}'.format('Ensemble Model Computational complexity: ', macs))
                results = evaluate_model(experiment_config['eval_type'],
                                        ensemble_model,
                                        config,
                                        test_train="train" in suffix)
                if pair_idxs[0] != 0:
                    results["Task A"], results["Task B"] = results["Task B"], results["Task A"]
                for idx,pair_idx in enumerate(pair_idxs):
                    results[f'Split {CONCEPT_TASKS[idx]}'] = pair[pair_idx]
                # results['Time'] = Merge.compute_transform_time
                results['Merging Fn'] = "ensemble"
                results['Model A'] = config['model']['name']
                results['Model B'] = config_b['model']['name']
                results.update(flatten_nested_dict(node_config, sep=' '))
                write_to_csv(results, csv_file=csv_file)
                print(results)
                all_results.append(results)
            # break
        # break
        # pdb.set_trace()
    # zero shortcut 'Joint': 0.1497, 'Per Task Avg': 0.2285, 'Task A': 0.1976, 'Task B': 0.2594
    print(f'Results of {node_config}: {all_results}')
    return results


if __name__ == "__main__":
    partial_layer = "all"
    stop_node_map = {7: 21, 13: 42, 19: 63, "all": None}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = args.config_name
    config_name_b = args.config_name_b
    skip_pair_idxs = []

    experiment_configs = []
    if "fs" in suffix:
        # for fr in [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]: # search more balanced factor
        # for fr in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91]: # search more balanced factor
        for fr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:

            experiment_configs.append({
                'stop_node': stop_node_map[partial_layer],
                'params': {
                    'a': 0.5,
                    'b': 1,
                    "fix_rate": fr
                }
            })
    else:
        experiment_configs.append({
            'stop_node': stop_node_map[partial_layer],
            'params': {
                'a': 0.5,
                'b': 1,
                "fix_rate": 0.5
            }
        })

    raw_config = get_config_from_name(config_name, device=device)
    raw_config_b = get_config_from_name(config_name_b, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    run_pairs = find_runable_pairs(model_dir,
                                   model_name,
                                   skip_pair_idxs=skip_pair_idxs)
    print(run_pairs)
    csv_file = os.path.join(
        './csvs', raw_config['dataset']['name'], raw_config['model']['name'],
        raw_config['eval_type'],
        f'heteromerging_cka_{config_name}_{config_name_b}_{partial_layer}_configurations{suffix}.csv')


    for node_config in experiment_configs:
        raw_config['dataset'].update(node_config.get('dataset', {}))
        run_node_experiment(node_config=node_config,
                            experiment_config=raw_config,
                            experiment_config_b=raw_config_b,
                            pairs=run_pairs,
                            csv_file=csv_file)
        gc.collect()
