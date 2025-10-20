import torch
from dataset import StockDataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
# from torch_geometric.nn import GCNConv,GATv2Conv
from sklearn.metrics import confusion_matrix
# from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter_softmax
import math
from einops import rearrange, repeat
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
# import qiskit
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from PatchTST_backbone import PatchTST_backbone, TSTEncoder
# from PatchTST_layers import series_decomp

# from RevIN import RevIN
from typing import Callable, Optional
from torch import Tensor


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual, drop_flag=0):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.drop_flag = drop_flag

    def forward(self, x, res):
        if self.residual:
            x = x + res
        x = self.norm(x)
        if self.drop_flag:
            x = self.dropout(x)
        return x
class Configs(object):
    ab = 0
    modes = 32
    mode_select = 'random'
    # version = 'Fourier'
    version = 'Wavelets'
    moving_avg = [12, 24]
    L = 1
    base = 'legendre'
    cross_activation = 'tanh'
    seq_len = 125
    label_len = 48
    pred_len = 1
    output_attention = True
    enc_in = 3
    dec_in = 3
    d_model = 16
    embed = 'timeF'
    dropout = 0.05
    freq = 'h'
    factor = 1
    n_heads = 8
    d_ff = 16
    e_layers = 2
    d_layers = 1
    c_out = 1
    activation = 'gelu'
    wavelet = 0
    fc_dropout = 0.2
    head_dropout = 0
    individual = 1
    patch_len = 7
    stride = 2
    padding_patch = 'end'
    revin = True
    affine = True
    subtract_last = True
    decomposition = 0
    kernel_size = 3

def calculate_mcc(y_true, y_pred):
    """
    计算 MCC (Matthew's Correlation Coefficient) 指标。
    
    Args:
        y_true (list or np.ndarray): 真实标签。
        y_pred (list or np.ndarray): 预测标签。
    
    Returns:
        float: MCC 值。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch[0]['time']) ==0:
        return None
    if len(batch) == 0:
        return None
    times = [item["time"] for item in batch]
    stock_features = [item["stock_features"] for item in batch]
    labels = [item["label"] for item in batch]
    # print(batch)
    comps = [item["comp"] for item in batch]
    texts = [item["texts"] for item in batch]
    return {
        "comp":comps,
        "time": times,
        "stock_features": torch.nn.utils.rnn.pad_sequence(stock_features, batch_first=True, padding_value=0),
        "texts": torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0),
        "label": torch.stack(labels)
    }
def calculate_mcc(y_true, y_pred):
    """
    计算 MCC (Matthew's Correlation Coefficient) 指标。
    
    Args:
        y_true (list or np.ndarray): 真实标签。
        y_pred (list or np.ndarray): 预测标签。
    
    Returns:
        float: MCC 值。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0
class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            
        self.encoder = TSTEncoder(10 ,3, 1,d_k= None,d_v= None,d_ff= 16,norm= 'BatchNorm',attn_dropout=0.0,dropout=0.05,activation='gelu',res_attention=True,n_layers=2,pre_norm=False,store_attn=False)
        self.ln = nn.Linear(3,1)
        # self.test_mamba = MambaTimeSeriesModel(3, 64, 2, True)
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x,z1 = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            x = self.ln(x)
        return x,z1

class TimeTextModel(nn.Module):
    def __init__(self, stock_feature_dim, hidden_dim, window_size):
        super(TimeTextModel,self).__init__()
        self.window_size = window_size
        self.stock_lstm = nn.LSTM(stock_feature_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.stock_lstm1 = nn.LSTM(stock_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.window_size = window_size
        embed = 300
        embedding = 'embedding_SougouNews.npz'
        # self.embedding_pretrained = torch.tensor(
        #     np.load('/home/zhaokx/legal_case_retrieval/law_case_retrieval_temp1/others/embedding_SougouNews.npz')["embeddings"].astype('float32')) \
        #     if embedding != 'random' else None  # 预训练词向量
        # embed = self.embedding_pretrained.size(1) \
        #     if self.embedding_pretrained is not None else 300
        # self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=True)
        self.embedding = nn.Embedding(186100,300)
        self.text_fc = nn.Linear(144, hidden_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        )
        self.mlp = nn.Sequential(nn.Linear(192,32),nn.ReLU(),nn.Linear(32,2))
        configs = Configs()
                # 一维卷积层
        self.conv = nn.Conv1d(
            in_channels=10 * hidden_dim,  # 合并 num_docs 和 hidden_dim
            out_channels=hidden_dim,  # 输出通道
            kernel_size=3,
            padding=3 // 2  # 保持长度不变
        )
        # 池化层
        self.pool = nn.MaxPool1d(2)
    def conv_and_pool(self, x, conv):
        """卷积 + 最大池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, stock_features, texts):
        texts = texts.squeeze(0)
        batch_size, num_docs, max_words = texts.size()
        texts = texts.view(-1, max_words)
        xtext = self.embedding(texts.to(device)).unsqueeze(1)
        text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
        text_repr2 = self.text_fc(text_repr)
        text_repr = text_repr2.view( 1, batch_size, num_docs, -1)
        text_repr = text_repr.view( 1,  batch_size, num_docs * text_repr2.shape[-1]).permute(0,2,1)  # [B, seq_length , hidden_dim * 
        output = self.conv(text_repr).permute(0,2,1)
        output = self.pool(output)
        stock_output, _ = self.stock_lstm(stock_features)

        segment_length = 5
        batch_size, total_length, stock_feat = stock_features.shape
    
        num_segments = total_length - segment_length + 1  # 滑动窗口的数量
    
        # 2. 为每个滑动窗口创建股票数据段
        stock_segments = []
        for i in range(num_segments):
            segment = stock_features[:, i:i+segment_length, :]  # [1, 5, 3]
            stock_segments.append(segment)
        
        stock_segments = torch.cat(stock_segments, dim=0)  # [num_segments, 5, 3]
        # 3. 使用LSTM处理每个股票段
        # LSTM前向传播 [num_segments, 5, 3] -> [num_segments, 5, hidden_size]
        lstm_out, (hidden, cell) = self.stock_lstm(stock_segments)
        stock_last_day_features = lstm_out[:, -1, :]
        combined_repr = torch.cat([stock_last_day_features.unsqueeze(0), output[:,4:,:]],dim=-1)
        output = self.mlp(combined_repr.squeeze(1))
        # print(output.shape)
        return output, output

def main():
    # dataset_train = StockDataset("/home/zhaokx/method2/dataset/acl18/tweet/preprocessed","/home/zhaokx/method2/dataset/acl18/price/preprocessed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/method2/dataset/acl18/tweet/preprocessed","/home/zhaokx/method2/dataset/acl18/price/preprocessed",mode='test')
    # dataset_train = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-CN/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-CN/price/preprocessed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-CN/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-CN/price/preprocessed",mode='test')
    # dataset_train = StockDataset("/home/zhaokx/multi-ts/temp204/CMIN-Dataset-main/beifen/stock_text_preCMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/processed",mode='train')
    # dataset_val = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/processed",mode='test')
    dataset_train = StockDataset("/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/news/preprocessed/","/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/price/preprocessed",mode='train')
    dataset_val = StockDataset("/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/news/preprocessed","/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/price/preprocessed",mode='val')
    dataset_test = StockDataset("/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/news/preprocessed","/root/autodl-tmp/dir1/causal-ts/my_code/NeuroDCG/dataset/CMIN-US/price/preprocessed",mode='test')
    batch_size = 1
    num_epochs = 100

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    stock_feature_dim = 3
    hidden_dim = 128
    window_size = 5

    model = TimeTextModel(stock_feature_dim, hidden_dim, window_size).to(device)
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCEWithLogitsLoss()
    best_eval = 0
    optimizer = optim.RAdam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        correct2 = 0
        total1 = 0
        batch_id = 0
        for batch in train_dataloader:
            if batch is None:
                continue
            stock_features = batch["stock_features"]
            labels = batch["label"][:,4:]
            texts = batch["texts"]
            # outputs, combined_repr, stock_output, pred, features_all, text_repr = model(stock_features.to(device),texts.to(device))
            pred, pred_text = model(stock_features.to(device),texts.to(device))
            # pred = pred.squeeze(-1).squeeze(-1)
            labels = labels.squeeze(0)
            # labesl1 = labels[-pred.shape[0]:]
            # + criterion1(torch.sigmoid(pred),labesl1.float().unsqueeze(1).to(device))
            optimizer.zero_grad()
            # criterion(pred.squeeze(0).squeeze(-1), labels.to(device))+ 
            loss = criterion(pred_text.squeeze(0).squeeze(-1), labels.to(device))
            # loss = criterion1(torch.sigmoid(pred.squeeze(0).squeeze(-1)),labels.float().to(device))
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(pred_text.squeeze(0), 1)
            # correct += (predicted == labels.to(device)).sum().item()
            correct2 += (labels.to(device) == predicted).sum().item()
            total += labels.size(0)
            # total1 += labesl1.size(0)
            batch_id+=1
        accuracy = correct2 / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        correct2 = 0
        total1 = 0
        pred_all = []
        label_all = []
        pred_all1 = []
        val_correct2 = 0
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                stock_features = batch["stock_features"]
                labels = batch["label"][:,4:]
                texts = batch["texts"]
                # outputs, combined_repr, stock_output, pred, features_all, text_repr = model(stock_features.to(device),texts.to(device))
                # outputs, pred, features_all = model(stock_features.to(device),texts.to(device))
                pred, pred_text = model(stock_features.to(device),texts.to(device))
                labels = labels.squeeze(0)
                # criterion(pred.squeeze(0).squeeze(-1), labels.to(device)) + 
                loss = criterion(pred_text.squeeze(0).squeeze(-1), labels.to(device))
                # pred = pred.squeeze(-1).squeeze(-1)
                # labesl1 = labels[-pred.shape[0]:]+ criterion1(torch.sigmoid(pred),labesl1.float().unsqueeze(1).to(device))
                # loss = criterion(outputs, labels.to(device)) 
                # loss = criterion1(torch.sigmoid(pred.squeeze(0).squeeze(-1)),labels.float().to(device))
                val_loss += loss.item()
                _, predicted = torch.max(pred_text.squeeze(0), 1)
                # val_correct += (predicted == labels.to(device)).sum().item()
                val_total += labels.size(0)
                # correct2 += (labesl1.to(device) == (torch.sigmoid(pred.squeeze(-1)) > 0.5).int()).sum().item()
                # total1 += labesl1.size(0)
                # pred_all.extend(predicted.tolist())
                label_all.extend(labels.tolist())
                pred_all1.extend(predicted.tolist())
                # print(predicted.shape,labels.shape)
                val_correct2 +=(labels.to(device) == predicted).sum().item()
        # mcc = calculate_mcc(pred_all,label_all)
        # val_accuracy = val_correct / val_total * 100
        val_acc1 = val_correct2 / val_total * 100
        if val_acc1 > best_eval:
            best_eval = val_acc1
            # torch.save(model,"model_text37.pth")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc1:.2f}%, Validation Mcc: {calculate_mcc(pred_all1,label_all):.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        correct2 = 0
        total1 = 0
        pred_all = []
        label_all = []
        pred_all1 = []
        val_correct2 = 0
        with torch.no_grad():
            for batch in test_dataloader:
                if batch is None:
                    continue
                stock_features = batch["stock_features"]
                labels = batch["label"][:,4:]
                texts = batch["texts"]
                # outputs, combined_repr, stock_output, pred, features_all, text_repr = model(stock_features.to(device),texts.to(device))
                # outputs, pred, features_all = model(stock_features.to(device),texts.to(device))
                pred, pred_text = model(stock_features.to(device),texts.to(device))
                labels = labels.squeeze(0)
                # criterion(pred.squeeze(0).squeeze(-1), labels.to(device)) + 
                loss = criterion(pred_text.squeeze(0).squeeze(-1), labels.to(device))
                # pred = pred.squeeze(-1).squeeze(-1)
                # labesl1 = labels[-pred.shape[0]:]+ criterion1(torch.sigmoid(pred),labesl1.float().unsqueeze(1).to(device))
                # loss = criterion(outputs, labels.to(device)) 
                # loss = criterion1(torch.sigmoid(pred.squeeze(0).squeeze(-1)),labels.float().to(device))
                val_loss += loss.item()
                _, predicted = torch.max(pred_text.squeeze(0), 1)
                # val_correct += (predicted == labels.to(device)).sum().item()
                val_total += labels.size(0)
                # correct2 += (labesl1.to(device) == (torch.sigmoid(pred.squeeze(-1)) > 0.5).int()).sum().item()
                # total1 += labesl1.size(0)
                # pred_all.extend(predicted.tolist())
                label_all.extend(labels.tolist())
                pred_all1.extend(predicted.tolist())
                # print(predicted.shape,labels.shape)
                val_correct2 +=(labels.to(device) == predicted).sum().item()
        # mcc = calculate_mcc(pred_all,label_all)
        # val_accuracy = val_correct / val_total * 100
        val_acc1 = val_correct2 / val_total * 100
        if val_acc1 > best_eval:
            best_eval = val_acc1
            # torch.save(model,"model_text37.pth")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc1:.2f}%, Validation Mcc: {calculate_mcc(pred_all1,label_all):.2f}%")
        # print(correct2/total1*100)
if __name__ == "__main__":
    main()