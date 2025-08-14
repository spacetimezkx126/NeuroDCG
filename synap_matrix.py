import os
import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, HeteroConv
from torch_geometric.data import Data, Batch,HeteroData
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel
from datetime import datetime
import torch.nn as nn
import random

class AntColonyOptimizer:
    def __init__(self, num_nodes, num_ants=10, max_iter=50, alpha=0.5, beta=0.5, rho=0.005):
        self.num_nodes = num_nodes
        self.num_ants = num_ants    
        self.max_iter = max_iter    
        self.alpha = alpha          
        self.beta = beta            
        self.rho = rho              
        self.pheromone = np.ones((num_nodes, num_nodes)) * 0.1
        
    def optimize(self, adj_matrix, heuristic_info):
        adj_matrix = adj_matrix.detach().cpu().numpy()
        """用蚁群算法寻找最优路径（最强因果关系链）"""
        best_path = None
        best_length = float('inf')
        
        for _ in range(self.max_iter):
            all_paths = []
            all_lengths = []
            for _ in range(self.num_ants):
                path = self._construct_path(adj_matrix, heuristic_info)
                length = self._evaluate_path(path, adj_matrix)
                all_paths.append(path)
                all_lengths.append(length)
                if length < best_length:
                    best_length = length
                    best_path = path
            self._update_pheromone(all_paths, all_lengths)
        
        return best_path, best_length
    
    def _construct_path(self, adj_matrix, heuristic_info, path_part = None):
        if path_part is not None:
            path = [path_part]
        else:
            path = [random.randint(0, self.num_nodes - 1)]
        unvisited = set(range(self.num_nodes)) - set(path)
        
        while unvisited:
            current = path[-1]
            probs = []
            for node in unvisited:
                if adj_matrix[current, node] > 0:
                    pheromone = self.pheromone[current, node]
                    heuristic = heuristic_info[current, node]
                    probs.append((node, pheromone*self.alpha + heuristic*self.beta))
                else:
                    probs.append((node, 0))
            total = sum(p[1] for p in probs)
            if total == 0:
                next_node = random.choice(list(unvisited))
            else:
                weights = [p[1]/total for p in probs]
                next_node = random.choices([p[0] for p in probs], weights=weights)[0]
            
            path.append(next_node)
            unvisited.remove(next_node)
        
        return path
    
    def _evaluate_path(self, path, adj_matrix):
        return sum(adj_matrix[path[i], path[i+1]] for i in range(len(path)-1))
    
    def _update_pheromone(self, all_paths, all_lengths):
        self.pheromone *= (1 - self.rho) 
        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path)-1):
                self.pheromone[path[i], path[i+1]] += 1.0 / length 
        # return self.pheromone
class Synap_Matrix(nn.Module):
    def __init__(self, last_step_dim) -> None:
        super(Synap_Matrix, self).__init__()
        self.aco = AntColonyOptimizer(
            num_nodes=5,
            num_ants=15, max_iter=20, alpha=0.5, beta=0.5, rho=0.0005
        
        )

    def forward(self, last_step_output, epoch = 0, path_part = None):
        row_id = 0
        col_id = 0
        sum1 = 0
        key_order = []
        num = 0
        for etype in last_step_output:
            if etype[-1] == 'price':
                sum1 += last_step_output[etype].shape[1]
                key_order.append(etype[0])
                num += 1
        init_matrix = torch.zeros((sum1,sum1))
        adj = torch.zeros((num,num)).to(last_step_output[('virtual','to','price')].device)
        r, c = 0, 0
        factors = []
        for key in last_step_output:
            if key[-1] == 'price':
                end_row_id = row_id + last_step_output[key].shape[-1]
                A = torch.mean(last_step_output[key],dim=0).unsqueeze(1)
                factors.append(key[0])
                
            for key1 in last_step_output: 
                if key[-1] == 'price' and key1[-1] == 'price':
                    if col_id == sum1:
                        col_id = 0
                        c = 0
                    end_col_id = col_id + last_step_output[key1].shape[1]
                    B = torch.mean(last_step_output[key1],dim=0).unsqueeze(1).permute(1,0)
                    C = abs(A - B)
                    init_matrix[row_id:end_row_id, col_id:end_col_id] = C
                    adj[r,c] = torch.mean(C)
                    col_id = end_col_id
                    c += 1
            row_id = end_row_id
            if key[-1] == 'price':
                r += 1
        heuristic_info = 1.0 / (adj + 1e-8) 
        if epoch !=0:
            best_path, best_length = self.aco.optimize(
            adj_matrix=adj,
            heuristic_info=heuristic_info
            )
        else:
            best_path = self.aco._construct_path(
            adj_matrix=adj,
            heuristic_info=heuristic_info,
            path_part = path_part
            )
        
        return init_matrix, best_path, factors