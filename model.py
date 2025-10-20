import os
import json
import torch

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
from struct_prompt import *
from synap_matrix import *

from torch_geometric.data import Data, Batch,HeteroData
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
from thop import profile, clever_format
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_scatter import scatter_mean
from collections import Counter
class GATSimConv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_j
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_k
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
            (default: :obj:`False`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        self.flow = 'source_to_target'
        self.trans = nn.Linear(1,1,bias=True)
        self.comb = nn.Linear(256,128)

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    # @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        factor_wei: Optional[int] = 1,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    # @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        factor_wei: Optional[int] = 1,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    # @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        factor_wei: Optional[int] = 1,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        factor_wei: Optional[int] = 1,
        return_attention_weights: Optional[bool] = None,
        
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")
        alpha = self.edge_updater(edge_index, x=(x_l, x_r),
                                  edge_attr=edge_attr)
        alpha = alpha * factor_wei
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias
        out = self.comb(torch.cat([out, x_r.squeeze(1)],dim=-1))
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, 
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x_i_norm = F.normalize(x_i, p=2, dim=-1)  # [num_edges, heads, in_channels]
        x_j_norm = F.normalize(x_j, p=2, dim=-1)
        
        cos_sim = (x_i_norm * x_j_norm).sum(dim=-1)  # [num_edges, heads]
        
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)  # [num_edges, heads * out_channels]
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)  # [num_edges, heads, out_channels]
            
            edge_weight = edge_attr.mean(dim=-1)  # [num_edges, heads]
            cos_sim = cos_sim * F.sigmoid(edge_weight) 
        
        alpha = self.trans(F.leaky_relu(cos_sim, 0.02))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
# 修改后的GATConv层，能够返回注意力权重
class GATv2ConvWithAlpha(GATv2Conv):
    def forward(self, x, edge_index, edge_attr, wei, return_alpha=False):
        out, (adj, alpha) = super().forward(x, edge_index, edge_attr, factor_wei = wei, return_attention_weights = return_alpha)
        if return_alpha:
            return out, alpha
        else:
            return out

class NeuroDCG(nn.Module):
    def __init__(self, metadata = None):
        super(NeuroDCG, self).__init__()
        in_channels = {
            'price': 128,
            'score': 128,
            'other': 128,
            'news': 128,
            'virtual': 128,
            'sector': 128,
            'keywords': 128
        }
        self.single = Struct_Prompt_TextCNN(in_channels, metadata = metadata)
        self.last_step_dim = {
            'price': 128,
            'score': 128,
            'other': 128,
            'news': 128,
            'virtual': 128,
            'sector': 128,
        }
        self.multi = Synap_Matrix(self.last_step_dim)
        # self.zero_prop = GATv2Conv(128,128, heads=1,edge_dim=64,add_self_loops=False)
        self.first_prop = GATSimConv(128,128, heads=1,edge_dim=64)
        # self.second_prop = GATSimConv(128,128, heads=1,edge_dim=64)
        # self.third_prop = GATSimConv(128,128, heads=1,edge_dim=64)
        # self.fourth_prop = GATSimConv(128,128,heads=1,edge_dim=64)
        # self.last_prop = GATSimConv(128,128, heads=1,edge_dim=64)
        self.hetero_gnn1 = HeteroConv(
            {
             ('first','to','second'): self.first_prop,
             ('second','to','third'): self.first_prop, 
             ('third','to','forth'): self.first_prop,
             ('forth','to','fifth'): self.first_prop,
             ('fifth','to','first'): self.first_prop,
            
            # ('first','to','second'): self.first_prop,
            # ('first','to','third'): self.first_prop,
            # ('first','to','forth'): self.first_prop,
            # ('first','to','fifth'): self.first_prop,
            #  ('second','to','third'): self.first_prop, 
            #  ('second','to','first'): self.first_prop,
            #  ('second','to','forth'): self.first_prop,
            #  ('second','to','fifth'): self.first_prop,
            #  ('third','to','forth'): self.first_prop,
            #  ('third','to','second'): self.first_prop,
            #  ('third','to','first'): self.first_prop,
            #  ('third','to','fifth'): self.first_prop,
            #  ('forth','to','fifth'): self.first_prop,
            #  ('forth','to','first'): self.first_prop,
            #  ('forth','to','second'): self.first_prop,
            #  ('forth','to','third'): self.first_prop,
            #  ('fifth','to','first'): self.first_prop,
            #  ('forth','to','forth'): self.first_prop,
            #  ('forth','to','second'): self.first_prop,
            #  ('forth','to','third'): self.first_prop,
            }, aggr='sum'
        )
        self.map = {
            0: "first",
            1: "second",
            2: "third",
            3: "forth",
            4: "fifth"
        }
        self.test_mul = nn.Sequential(nn.Linear(320+128+64,64+32),nn.ReLU(),nn.Linear(64+32,1))
        self.path = [0,1,2,3,4]
        self.all_path = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        self.factors = []
        self.node_types = ['first', 'second', 'third', 'forth', 'fifth']
        self.attn_weights = torch.nn.Parameter(torch.ones(len(self.node_types)))
        self.map_path = {}
        self.epoch = -1
    def forward(self, data, epoch = 0):

        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr
            for edge_type in data.edge_types 
            if hasattr(data[edge_type], 'edge_attr')
        }

        output, total_rank, ranks, first_dict_con, text_emb, alpha_con, target_dict, test , edge_dict,  simu_all, grad_diff_all, edge_index_news, selected_edge_index_news, edge_index_keywords, selected_edge_index_keywords = self.single(data.x_dict, data.edge_index_dict, edge_attr_dict)
        if len(self.factors) == 0:
            for key in first_dict_con:
                self.factors.append(key[0])
            matrix, path, factors = self.multi(first_dict_con, epoch)
        if epoch != 'e':
            matrix, path, factors = self.multi(first_dict_con, epoch)
            self.path = path
            if self.epoch != epoch:
                self.map_path = {}
                self.epoch = epoch
            if "*".join([str(b) for b in path]) in self.map_path:
                self.map_path["*".join([str(b) for b in path])] += 1
            else:
                self.map_path["*".join([str(b) for b in path])] = 1
        else:
            max_key = max(self.map_path, key=lambda k: self.map_path[k])
            path = [int(b) for b in max_key.split("*")][0]
            matrix, path, factors = self.multi(first_dict_con, path_part = path)
        # path = self.path
        factors = self.factors
        dict_hetero = {}
        edge_index_dict = {}
        for key in first_dict_con:
            if key[-1] == 'price':
                dict_hetero[key[0]] = first_dict_con[key]
        dict_hetero['price'] = target_dict['price']
        # for k in range(len(self.all_path)):
        #     # for i in len(self.all_path[k]):
        #     #     if i == 1:
        #     #         break
        #     dict_hetero[self.map[self.all_path[k][0]]] = dict_hetero[factors[self.all_path[k][0]]]
        #     dict_hetero[self.map[self.all_path[k][1]]] = dict_hetero[factors[self.all_path[k][1]]]
        #     sources = list(range(len(dict_hetero[factors[self.all_path[k][0]]])))
        #     targets = list(range(len(dict_hetero[factors[self.all_path[k][1]]])))
        #     edge_index_dict[self.map[self.all_path[k][0]],'to',self.map[self.all_path[k][1]]] = torch.tensor(list(zip(sources, targets))).permute(1,0).to(dict_hetero[self.map[self.all_path[k][0]]].device)
        #     edge_index_dict[self.map[self.all_path[k][1]],'to',self.map[self.all_path[k][0]]] = torch.tensor(list(zip(targets, sources))).permute(1,0).to(dict_hetero[self.map[self.all_path[k][0]]].device)
        #     # print(edge_index_dict[self.map[self.all_path[k][0]],'to',self.map[self.all_path[k][1]]])
        for i in range(len(path)):
            dict_hetero[self.map[i]] = dict_hetero[factors[path[i]]]
            sources = list(range(len(dict_hetero[factors[path[i]]])))
            targets = list(range(len(dict_hetero[factors[path[i+1 if i+1 < len(path) else 0]]])))
            edge_index_dict[self.map[i],'to',self.map[i+1 if i+1 < len(path) else 0]] = torch.tensor(list(zip(sources, targets))).permute(1,0).to(dict_hetero[self.map[i]].device)
        sources = list(range(len(dict_hetero[factors[path[-1]]])))
        targets = list(range(len(dict_hetero[factors[path[-1]]])))
        x_dict = self.hetero_gnn1(dict_hetero, edge_index_dict)
        # inputs = (x_dict, edge_index_dict)
    
        # # # 使用thop进行profile
        # macs, params = profile(self.hetero_gnn1, inputs=inputs, verbose=True)
        
        # # # # 格式化输出
        # macs, params = clever_format([macs, params], "%.3f")
        
        # print(f"\n计算结果:")
        # print(f"MACs: {macs}")
        # print(f"参数数量: {params}")
        type_reprs = []
        for i, node_type in enumerate(self.node_types):
            if node_type in x_dict:
                x = x_dict[node_type]  # [num_nodes, hidden_dim]
            else:
                x = dict_hetero[node_type]
            type_reprs.append(x)
        
        attn = F.softmax(self.attn_weights, dim=0)  # [5]
        graph_repr = sum(w * repr for w, repr in zip(attn, type_reprs))
        out_cls = self.test_mul(torch.cat([dict_hetero['price'],test['price'],text_emb,graph_repr],dim=-1))
        return out_cls, total_rank, ranks, output, None, self.map_path, simu_all, grad_diff_all, edge_index_news, selected_edge_index_news, edge_index_keywords, selected_edge_index_keywords
        # return output, None, None, output, None, None, None, None