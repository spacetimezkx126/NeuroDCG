import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, Linear
from torch_geometric.data import Data, Batch,HeteroData
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel
from datetime import datetime

import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
import numpy as np
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_scatter import scatter

from typing import List
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import NodeType, EdgeType, Metadata, Adj, OptTensor
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import softmax
from typing import Dict, Union, Tuple, List, Optional

def group(outs: List[Tensor]) -> Tuple[Tensor, Tensor]:
    if len(outs) == 0:
        return None, None
    out = torch.stack(outs, dim=1)  # [N, M, out_channels]
    return out.sum(dim=1), None

class SynapticConvLayer(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,** kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.empty(1, out_channels))
        self.proj = nn.ModuleDict()
        for node_type, in_ch in self.in_channels.items():
            self.proj[node_type] = nn.Linear(in_ch, out_channels)
        self.lin_src = nn.ParameterDict() 
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_key = '__'.join(edge_type)
            
            self.lin_src[edge_key] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[edge_key] = nn.Parameter(torch.empty(1, heads, dim))


        self.grad_map = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.Sigmoid()
        )
        self.register_buffer('scale_factor', torch.tensor(1.0)) 
        self.map_loss = 0
        self.receptivity = nn.ModuleDict()
        for node_type in metadata[0]:
            self.receptivity[node_type] = nn.Linear(out_channels, 1) 

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)
        for param in self.receptivity.values():
            reset(param)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_feedback: bool = True, 
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor], Dict[EdgeType, Tensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}
        for node_type, x in x_dict.items():
            if node_type in self.in_channels:
                x_proj = self.proj[node_type](x).view(-1, H, D)
                if node_type in [et[2] for et in edge_index_dict.keys()]:
                    x_proj = x_proj.requires_grad_(True)
                x_node_dict[node_type] = x_proj
                out_dict[node_type] = []
        feedback_dict = {}
        alpha_wei = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type in self.metadata[1]:
                src_type, _, dst_type = edge_type
                edge_key = '__'.join(edge_type)
                lin_src = self.lin_src[edge_key]  # [1, H, D]
                lin_dst = self.lin_dst[edge_key]  # [1, H, D]
                x_src = x_node_dict[src_type].clone()  # [N_src, H, D]
                x_dst = x_node_dict[dst_type].clone()  # [N_dst, H, D]
                alpha_src = (x_src * lin_src).sum(dim=-1)  # [N_src, H]
                alpha_dst = (x_dst * lin_dst).sum(dim=-1)  # [N_dst, H]
                x_dst_before = x_dst.detach().clone() 
                out_propagate = self.propagate(
                    edge_index,
                    x=(x_src, x_dst),
                    alpha=(alpha_src, alpha_dst),
                    receptivity=torch.ones(x_dst.shape[0], 1, device=x_dst.device)  
                )
                out_propagate = out_propagate.view(-1,H,D)
                loss_diff = F.mse_loss(out_propagate, x_dst_before) 
                if self.training:
                    grad_diff = torch.autograd.grad(
                        loss_diff,
                        x_dst,
                        retain_graph=True
                    )[0]
                    feat_diff = out_propagate - x_dst_before  # [N_dst, H, D]
                    feat_diff_flat = feat_diff.view(-1, H*D)  # [N_dst, H*D]
                    grad_diff_flat = grad_diff.view(-1, H*D)  # [N_dst, H*D]
                    map_output = self.grad_map(feat_diff_flat)  # [N_dst, H*D]
                    simu = map_output * feat_diff_flat
                    grad_diff1 = simu.view(-1, H, D)
                else:
                    feat_diff = out_propagate - x_dst_before  # [N_dst, H, D]
                    feat_diff_flat = feat_diff.view(-1, H*D)  # [N_dst, H*D]
                    map_output = self.grad_map(feat_diff_flat)  # [N_dst, H*D]
                    grad_diff_approx = map_output * feat_diff_flat
                    grad_diff1 = grad_diff_approx.view(-1, H, D)
                grad_diff = torch.mean(grad_diff1,dim=-1)
                if grad_diff.ndim == 1:
                    grad_diff = grad_diff.unsqueeze(-1)
                grad_norm = grad_diff.norm(dim=1,p=2, keepdim=True)  # [N_dst, 1]
                grad_diff = grad_diff / (grad_norm + 1e-8)
                receptivity = grad_diff
                out = self.propagate(
                    edge_index,
                    x=(x_src, x_dst),
                    alpha=(alpha_src, alpha_dst),
                    receptivity=receptivity, 
                )
                if edge_type[2] == 'price' or edge_type == ('keywords','to','virtual'):
                    feedback_dict[edge_type] = receptivity
                out = F.relu(out)
                out_dict[dst_type].append(out)
                alpha_wei[edge_type] = alpha_src.squeeze(1)
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = group(outs)
            out_dict[node_type] = out if out is not None else x_node_dict[node_type].view(-1, self.out_channels)
            semantic_attn_dict[node_type] = attn
        if self.training:
            if return_feedback:
                return out_dict, feedback_dict, alpha_wei, simu, grad_diff_flat
            return out_dict, alpha_wei, simu, grad_diff_flat
        else:
            if return_feedback:
                return out_dict, feedback_dict, alpha_wei, None, None
            return out_dict, alpha_wei, None, None

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                receptivity: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        receptivity_j = receptivity[index].view(-1, 1)
        alpha = alpha_j + alpha_i 
        alpha = alpha * receptivity_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)  # [E, H, D]
        return out.view(-1, self.out_channels)  # [E, out_channels]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')


def get_batch_topk_ranks(scores, top_k):
    top_values, top_indices = torch.topk(scores, top_k, dim=1)
    top_ranks = torch.arange(1, top_k + 1, device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)
    return top_values, top_indices, top_ranks

def accumulate_target_ranks(batch_data, target=0):
    batch_data = torch.cat([batch_data,torch.ones(batch_data.shape[0],1).to(batch_data.device)* target],dim=1)
    batch_size, num_elements = batch_data.shape
    target_mask = (batch_data == target).float()
    target_indices = torch.argmax(target_mask, dim=1)  # 形状 [batch_size]
    total_rank = target_indices.sum()
    return total_rank, target_indices

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SourceExpertMoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim     
        self.hidden_dim = hidden_dim    
        self.output_dim = output_dim    
        self.num_experts = num_experts  
        self.top_k = top_k              

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim * num_experts, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, features_list: List[torch.Tensor], receptivity: torch.Tensor):
        count = {}

        M = len(features_list)
        N = features_list[0].shape[0]
        assert M == self.num_experts

        features_per_source = torch.stack(features_list, dim=1)  # [N, M, input_dim]
        features_stacked = torch.cat(features_list, dim=1)  # [N, M*input_dim]
        gate_logits = self.gate(features_stacked)  # [N, M]
        gate_weights = F.softmax(gate_logits, dim=1)  # [N, M]
        top_k_weights, top_k_indices, top_ranks = get_batch_topk_ranks(gate_weights+torch.stack(receptivity,dim=1).squeeze(-1), self.top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True) 
        batch_indices = torch.arange(N, device=features_stacked.device).repeat_interleave(self.top_k)  # [N*k]
        expert_indices = top_k_indices.flatten()  # [N*k]
        batch_input = torch.gather(
            features_per_source[batch_indices],  # [N*k, M, input_dim]
            dim=1,
            index=expert_indices.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.input_dim)  # [N*k, 1, input_dim]
        ).squeeze(1)  # [N*k, input_dim]
        expert_outputs = torch.zeros(N * self.top_k, self.output_dim, device=features_stacked.device)
        ranks_all = [None] * self.num_experts
        total_all = [0] * self.num_experts
        for e in range(self.num_experts):
            mask = (expert_indices == e)
            total, ranks = accumulate_target_ranks(top_k_indices,e)
            if ranks_all[e] is None:
                ranks_all[e] = ranks
            else:
                ranks_all[e] = torch.cat([ranks_all[e],ranks],dim=0)
            total_all[e] += total
            if not mask.any():
                continue
            expert_outputs[mask] = self.experts[e](batch_input[mask])  # [num_mask, output_dim]
        per_sample_out = expert_outputs.view(N, self.top_k, self.output_dim)  # [N, k, output_dim]
        final_out = (per_sample_out * top_k_weights.unsqueeze(-1)).sum(dim=1)  # [N, output_dim]
        return final_out, ranks_all, total_all

class StepHeteroProcessor(nn.Module):
    def __init__(self, in_channels: dict, out_channels: int, metadata: Metadata, steps: List[dict], num_moe_experts: int = 4):
        super().__init__()
        self.steps = steps
        self.synaptic_layer = SynapticConvLayer(in_channels, out_channels, metadata)
        self.moe_layer = SourceExpertMoE(
            input_dim = 128, hidden_dim = 128, output_dim = 128, 
            num_experts= 5, top_k = 5

        )

    def forward(self, x_dict: Dict[NodeType, torch.Tensor], edge_index_dict: Dict[EdgeType, torch.Tensor]):
        x = x_dict.copy()
        receptivity_all = {}
        step_output = {}
        alpha_all = {}
        simu_all = torch.zeros((0,128)).to(x_dict[list(x_dict.keys())[0]])
        grad_diff_all = torch.zeros((0,128)).to(x_dict[list(x_dict.keys())[0]])
        for step in self.steps:
            if 'parallel' in step:
                parallel_edges = step['parallel']
                sub_edge_index = {et: ei for et, ei in edge_index_dict.items() if et in parallel_edges}
                out_dict, feedback_dict, alpha, simu, grad_diff = self.synaptic_layer(x, sub_edge_index)
                if simu is not None:
                    simu_all = torch.cat([simu_all,simu],dim=0)
                    grad_diff_all = torch.cat([grad_diff_all,grad_diff],dim=0)
                x.update(out_dict)
                for edge_type, feedback in feedback_dict.items():
                    src_type = edge_type[0]
                    edge_index = edge_index_dict[edge_type]
                    src_nodes = edge_index[0]
                    receptivity_all[edge_type] = feedback_dict[edge_type]
                    alpha_all[edge_type] = alpha[edge_type]
            elif 'aggr' in step:
                aggr_edges = step['aggr']
                target_type = aggr_edges[0][2]
                features_list = []
                recep_list = []
                for et in aggr_edges:
                    if et[2] != target_type:
                        continue
                    sub_edge_index = {et: edge_index_dict[et]}
                    out_dict, feedback_dict, alpha, simu, grad_diff = self.synaptic_layer(x, sub_edge_index)
                    features_list.append(out_dict[target_type])
                    recep_list.append(feedback_dict[et])
                    receptivity_all[et] = feedback_dict[et]
                    step_output[et] = out_dict[target_type]
                    alpha_all[et] = alpha[et]
                    if simu is not None:
                        simu_all = torch.cat([simu_all,simu],dim=0)
                        grad_diff_all = torch.cat([grad_diff_all,grad_diff],dim=0)
                aggregated, ranks, total_rank = self.moe_layer(features_list, recep_list)
                x[target_type] = aggregated

        return x, receptivity_all, total_rank, ranks, step_output,alpha_all, simu_all, grad_diff_all


class EdgeSelector(torch.nn.Module):
    def __init__(self, edge_types, initial_keep_thread=0.01, initial_keep_ratio = 0.85):
        super().__init__()
        self.keep_threads = torch.nn.ParameterDict({
            str(edge_type): torch.nn.Parameter(torch.tensor(initial_keep_thread))
            for edge_type in edge_types
        })
        self.initial_keep_ratio = torch.nn.ParameterDict({
            str(edge_type): torch.nn.Parameter(torch.tensor(initial_keep_ratio))
            for edge_type in edge_types
        })
        self.sel_edge_types = edge_types
    
    def forward(self, edge_index_dict, alpha_dict, edge_attr_dict):
        selected_edge_index = {}
        selected_edge_attr = {}
        selected_alpha = {}
        
        for edge_type in edge_index_dict:
            if edge_type not in self.sel_edge_types:
                selected_edge_index[edge_type] = edge_index_dict[edge_type]
                selected_edge_attr[edge_type] = edge_attr_dict[edge_type]
                continue
            
            alpha = alpha_dict[edge_type]
            keep_thread = self.keep_threads[str(edge_type)]
            
            quantiles = torch.quantile(alpha, q=1-self.initial_keep_ratio[str(edge_type)])
            mask = alpha > quantiles
            selected_edge_index[edge_type] = edge_index_dict[edge_type][:, mask]
            selected_alpha[edge_type] = alpha[mask]
            selected_edge_attr[edge_type] = edge_attr_dict[edge_type][mask]
        
        return selected_edge_index, selected_alpha, selected_edge_attr
    
    def get_keep_ratios(self):
        return {edge_type: torch.sigmoid(ratio).item() 
                for edge_type, ratio in self.keep_ratios.items()}

class Struct_Prompt_TextCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, heads=1, factor_num = 5, initial_wei=[1.0,1.0,1.0,1.0,1.0,1.0], metadata = None):
        super(Struct_Prompt_TextCNN, self).__init__()
        edge_types = [
        ('score', 'to', 'price'),
        ('other', 'to', 'price'),
        ('news', 'to','price'),
        ('virtual', 'to','price'),
        ('sector','to','price'),
        ('keywords','to','virtual')
        ]
        self.edge_types = edge_types
        self.para = nn.Parameter(torch.tensor(initial_wei))
        self.para2 = nn.Parameter(torch.tensor(initial_wei))
        self.para3 = nn.Parameter(torch.tensor(initial_wei))
        self.wei_ln = nn.Linear(factor_num, factor_num)
        self.edge_selector = EdgeSelector(edge_types)
        self.classifier = nn.Linear(hidden_channels*2, 1)
        self.classifier_con = nn.Linear(hidden_channels, 1)
        self.price_encoder = nn.LSTM(3, in_channels['price'], 1, batch_first=True, bidirectional=False)
        self.score_encoder = nn.Linear(1, in_channels['score'])
        self.other_encoder = nn.Linear(128*6, in_channels['other'])
        self.edge_attr_enc = nn.Linear(1,64)

        self.conv = nn.Conv1d(
            in_channels= 128*5,  
            out_channels= 128,  
            kernel_size=3,
            padding=3 // 2  
        )
        embed = 384
        self.embedding = nn.Embedding(186100, embed)
        self.conv1 = nn.Conv1d(
            in_channels= 128*5,  
            out_channels= 128,  
            kernel_size=3,
            padding=3 // 2 
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        )
        self.text_fc = nn.Linear(144, 128)
        self.pool = nn.MaxPool1d(2)
        self.mlp = nn.Sequential(nn.Linear(128+64+128,128),nn.ReLU(),nn.Linear(128,1))
        self.mapping_words = {}
        self.text_type = ['other','keywords','news','virtual','sector']
        self.metadata = metadata
        self.in_channels = in_channels
        steps = [
            {'parallel': [('keywords', 'to', 'virtual')]},{'parallel':[('price', 'rev_to', 'score'),('price', 'rev_to', 'news'),('price', 'rev_to', 'virtual'),('price', 'rev_to', 'other'),('price', 'rev_to', 'sector')]},
            {'aggr': [ ('score', 'to', 'price'), ('news', 'to', 'price'), ('virtual', 'to', 'price'),('other','to','price'),('sector','to','price')]}
        ]
        self.steppro = StepHeteroProcessor(
            in_channels=self.in_channels,
            out_channels=128,
            metadata=self.metadata,
            steps=steps,
            num_moe_experts=5
        )
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def embedding_texts(self,x_dict):
        embedding_texts = {}
        for type1 in self.text_type:
            max_word = self.mapping_words[type1]
            dict_size = x_dict[type1].size()
            if x_dict[type1].ndim == 3:
                batch_size ,num_docs ,max_word1 = dict_size
                assert max_word1 == max_word
            elif x_dict[type1].ndim == 2:
                texts_num , max_word1 = dict_size
                assert max_word1 == max_word
                batch_size = texts_num 
                num_docs = 1
            texts = x_dict[type1].reshape(-1, max_word)
            xtext = self.embedding(texts.to(x_dict[type1]).long()).unsqueeze(1)
            text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
            text_repr2 = self.text_fc(text_repr)
            text_repr = text_repr2.view(batch_size, num_docs, -1)
            text_repr = text_repr.view(1, batch_size, num_docs * text_repr2.shape[-1]).permute(0,2,1)
            embedding_texts[type1] = text_repr.permute(0,2,1).squeeze(0)
        return embedding_texts

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, epoch = 0):

        if x_dict['score'].ndim == 1:
            x_dict['score'] = x_dict['score'].unsqueeze(1)
        
        
        if x_dict['news'].ndim == 3:
           x_dict['news'] = x_dict['news'].squeeze(1)

        if len(list(self.mapping_words.keys())) == 0:
            for node_type in x_dict:
                if node_type in self.text_type:
                    self.mapping_words[node_type] = x_dict[node_type].shape[-1]

        embedded_text = self.embedding_texts(x_dict)
        x_dict = {
            'score': self.score_encoder(x_dict['score']) if x_dict['score'].shape[1]==1 else self.score_encoder(x_dict['score'].permute(1,0)),
            'other': self.other_encoder(embedded_text['other'].view(embedded_text['other'].shape[0],-1)),
            'news': embedded_text['news'],
            'virtual': embedded_text['virtual'],
            'sector': embedded_text['sector'],
            'keywords': embedded_text['keywords'],
            'price': self.price_encoder(x_dict['price'])[0][:,-1,:],
            'news_token': x_dict['news_token'][:,:5,:]
        }
        test, receptivity_all, total_rank, ranks, step_output, alpha_all, simu_all, grad_diff_all = self.steppro(x_dict, edge_index_dict)
        max_word = 40
        batch_size, num_docs, max_word = x_dict['news_token'].size()
        texts = x_dict['news_token'].reshape(-1, max_word)
        xtext = self.embedding(texts.to(x_dict['news_token'])).unsqueeze(1)
        text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
        text_repr2 = self.text_fc(text_repr)
        text_repr = text_repr2.view(batch_size, num_docs, -1)
        text_repr = text_repr.view(1, batch_size, num_docs * text_repr2.shape[-1]).permute(0,2,1)
        output = self.conv(text_repr).permute(0,2,1)
        output_temp = self.pool(output)


        rest = list(set(list(range(int(torch.max(edge_index_dict[('sector', 'to', 'price')][1]))+1))) - set([b.item() for b in edge_index_dict[('sector', 'to', 'price')][1]]))
        min_result = scatter(
            edge_index_dict[('sector', 'to', 'price')][0],  
            edge_index_dict[('sector', 'to', 'price')][1],  
            dim=0,
            dim_size=x_dict['price'].shape[0], 
            reduce='min'  
        )
        max_result = scatter(
            edge_index_dict[('sector', 'to', 'price')][0], 
            edge_index_dict[('sector', 'to', 'price')][1],
            dim=0,
            dim_size=x_dict['price'].shape[0],
            reduce='max' 
        )
        
        news_plus_one = torch.cat([x_dict['sector'],torch.zeros((1,x_dict['sector'].shape[1])).to(x_dict['news'].device)],dim=0)
        min_max_all = torch.cat([min_result.unsqueeze(1),max_result.unsqueeze(1)],dim=1)
        index_all = torch.cat([torch.tensor(list(range(int(b[0]),min(int(b[1])+1,int(b[0])+5)))+[news_plus_one.shape[0]-1]*(5-(int(b[1])+1-int(b[0])))).unsqueeze(0) for b in min_max_all],dim=0)
        num_docs = 5
        # bert
        padding_news = news_plus_one[index_all.flatten()]
        padding_news1 = padding_news.view(-1,5,padding_news.shape[-1]).flatten(1).unsqueeze(-1).permute(2,1,0)
        output1 = self.conv1(padding_news1).permute(0,2,1)
        output1 = self.pool(output1)
        combined_repr = torch.cat([x_dict['price'].unsqueeze(1),output_temp.permute(1,0,2),test['price'].unsqueeze(1)],dim=-1)
        output = self.mlp(combined_repr.squeeze(1))

        return output, total_rank, ranks, step_output, output_temp.permute(1,0,2).squeeze(1), None, x_dict, test, None, simu_all, grad_diff_all

