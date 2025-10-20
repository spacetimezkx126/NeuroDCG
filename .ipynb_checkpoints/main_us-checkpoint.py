import os
import json
import torch
import argparse
import pickle as pkl
import numpy as np
import copy
import pickle
import gzip

import torch.optim as optim
from torch.nn import functional as F
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, HeteroConv
from torch_geometric.data import Data, Batch,HeteroData
from read_data import read_cmin
from read_data import load_price
from model import *

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--device', type=int, default=1, help='Which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden GNN dimension')
parser.add_argument('--prompt_token_dim', type=int, default=128, help='Prompt token dimension')
parser.add_argument('--prompt_token_num', type=int, default=10, help='Number of prompt tokens')
parser.add_argument('--news_path', type=str, default='./dataset/us_score', help='Path to news data')
parser.add_argument('--price_path', type=str, default='./dataset/CMIN-US/price/preprocessed', help='Path to price data')
args = parser.parse_args()

def calculate_mcc(y_true, y_pred, mode = 'b'):
    """
    calculate MCC (Matthew's Correlation Coefficient) evaluation metric。
    
    Args:
        y_true (list or np.ndarray): true labels。
        y_pred (list or np.ndarray): prediction results。
    
    Returns:
        float: MCC value。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    
    denominator = np.sqrt(np.float128(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    return numerator / denominator if denominator != 0 else 0.0



class ProcessedDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, ds_name = 'CMIN-US', news_path = './dataset/us_score', price_path = './dataset/CMIN-US/price/preprocessed', mode = 'load', data_num = 'all'):
        super().__init__(root=None, transform=transform)

        self.cpu_device = torch.device("cpu")
        self.mode = mode
        self.data_cn_list = []
        data_period = {
            "CMIN-CN": {
                "train_start": "2018-01-01",
                "train_end": "2021-04-30",
                "val_start": "2021-05-01",
                "val_end": "2021-08-31",
                "test_start": "2021-09-01",
                "test_end": "2021-12-31"
            },
            "CMIN-US": {
                "train_start": "2018-01-01",
                "train_end": "2021-04-30",
                "val_start": "2021-05-01",
                "val_end": "2021-08-31",
                "test_start": "2021-09-01",
                "test_end": "2021-12-31"
            }
        }

        self.period = data_period[ds_name]
        if self.mode == 'all':
            all_news_data = read_cmin(news_path)
            # print(all_news_data)
            all_price_data, label = load_price(price_path)
            self.all_price = all_price_data
            self.all_label = label
            all_date = []
            company_list = []
            samples = {}
            all_companies = all_news_data.keys()
            for company in all_companies:
                if company != '.ipynb_checkpoints':
                    company_list.append(company)
                    news_by_date = all_news_data[company]
                    
                    prices_by_date = all_price_data[company]
                    samples[company] = []
                    for date_json_file in news_by_date.keys():
                        date_str = date_json_file.split('.')[0]
                        samples[company].append(date_str)
                        if date_str not in all_date:
                            all_date.append(date_str)
                    for date_json_file in prices_by_date.keys():
                        date_str = date_json_file
                        samples[company].append(date_str)
                        if date_str not in all_date:
                            all_date.append(date_str)
                    samples[company].sort()
            all_date.sort()
            self.all_news_data = all_news_data
            self.all_date = all_date
            self.all_date_inverse = {k:all_date[k] for k in range(len(all_date))}
            self.period_id = {}
            self.period_id['tr_st_id'] = self.all_date.index(self.period["train_start"])
            self.period_id['tr_ed_id'] = self.all_date.index(self.period["train_end"])
            self.period_id['val_st_id'] = self.all_date.index(self.period["val_start"])
            self.period_id['val_ed_id'] = self.all_date.index(self.period["val_end"])
            self.period_id['te_st_id'] = self.all_date.index(self.period["test_start"])
            self.period_id['te_ed_id'] = self.all_date.index(self.period["test_end"])
            self.hetero_graphs_train = []
            self.hetero_graphs_val = []
            self.hetero_graphs_test = []
            self.comp_names = []
            self.already = {}
            self.already_comp = []
            dict1 = './dict/dict_us.pkl'
            # dict1 = './dict/vocab_us.pkl'
            self.vocab = pkl.load(open(dict1,'rb'))
            self.count_hete = 0
            self.metadata = None
        if self.mode == 'all':
            self.hetero_graphs = self._build_hetero_graphs_from_json()
        elif self.mode == 'load':
            with gzip.open("hetero_data_us_train.pkl.gz", "rb") as f:
                self.hetero_graphs_train = pickle.load(f)
            with gzip.open("hetero_data_us_val.pkl.gz", "rb") as f:
                self.hetero_graphs_val = pickle.load(f)
            with gzip.open("hetero_data_us_test.pkl.gz", "rb") as f:
                self.hetero_graphs_test = pickle.load(f)
            self.sample_data = self.load_sample_meta()
            self.metadata = self.sample_data.metadata()
    def load_sample_meta(self):
        hetero_data = HeteroData()
        data_id = 100
        current_start = {
            'score': 0,
            'other': 0,
            'keywords': 0,
            'news': 0,
            'virtual': 0,
            'sector': 0
        }
        price = torch.zeros((3,3))
        label = torch.zeros((1))
        hetero_data['price'].x = price.unsqueeze(0)
        hetero_data['y'].x = label
        hetero_data['score'].x = torch.randn((0,10))
        hetero_data['other'].x = torch.randn((0,6,10)).long()
        hetero_data['keywords'].x = torch.randn((0,15)).long()
        hetero_data['news'].x = torch.randn((0,40)).long()
        hetero_data['virtual'].x = torch.randn((0,5)).long()
        hetero_data['sector'].x = torch.randn((0,10)).long()
        hetero_data['news_token'].x = torch.zeros((1,8,40)).long()
        hetero_data['score_pad'].x = torch.zeros((1,8,10)).long()

        hetero_data['keywords','to','virtual'].edge_index = torch.randn((2,0)).long()
        hetero_data['keywords','to','virtual'].edge_attr = torch.randn((0,1))
        hetero_data['sector','to', 'score'].edge_index = torch.randn((2,0)).long()
        hetero_data['sector','to', 'score'].edge_attr = torch.randn((0,1))
        hetero_data['score','to', 'sector'].edge_index = torch.randn((2,0)).long()
        hetero_data['score','to', 'sector'].edge_attr = torch.randn((0,1))

        all_keys = ['score','other','news','virtual','sector']
        for key in all_keys:
            hetero_data[key,'to','price'].edge_index = torch.randn((2,0)).long()
            hetero_data[key,'to','price'].edge_attr = torch.randn((0,1))
        hetero_data = ToUndirected()(hetero_data)
        return hetero_data
        
    def _tokenize_and_pad_text(self, text,max_seq_len = 40):
        indices = [self.vocab.get(char, self.vocab["<unk>"]) for char in text.split(" ")]
        padded = indices[:max_seq_len] + [self.vocab["<pad>"]] * max(0, max_seq_len - len(indices))
        # print(padded)
        return padded

    def _pad_texts_to_equal_length(self, tokenized_texts_org, max_docs = 5,max_seq_len = 40):
        empty_text = [self.vocab["<pad>"]] * max_seq_len # 单个空文本的填充值
        tokenized_texts = copy.deepcopy(tokenized_texts_org)
        if len(tokenized_texts) < max_docs:
            tokenized_texts.extend([empty_text] * (max_docs - len(tokenized_texts)))
        else:
            tokenized_texts = tokenized_texts[:max_docs]
        return tokenized_texts
    
    def get_news_token(self, comp, date_ind):
        all_news = self.all_news_data
        if self.all_date[date_ind] in all_news[comp]:
            # print(all_news[comp][self.all_date[date_ind]])
            # print(comp,self.all_date[date_ind])
            return [b['original_text'] if 'original_text' in b else (b['orginal_text'] if 'orginal_text' in b else "") for b in all_news[comp][self.all_date[date_ind]][0]]
            # [self.all_date[date_ind]]
        return None
    
    def get_other_features(self, comp, date_ind):
        all_news = self.all_news_data
        if self.all_date[date_ind] in all_news[comp]:
            keywords = [b['features']['keywords'] if 'features' in b and 'keywords' in b['features'] else [] for b in all_news[comp][self.all_date[date_ind]][0]]
            sectors = [b['features']['sector_focus'] if 'features' in b and 'sector_focus' in b['features'] else [] for b in all_news[comp][self.all_date[date_ind]][0]]
            Correlation = [[b['scores']['Correlation']] if 'scores' in b and 'Correlation' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Sentiment = [[b['scores']['Sentiment']] if 'scores' in b and 'Sentiment' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Importance = [[b['scores']['Importance']] if 'scores' in b and 'Importance' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Impact = [[b['scores']['Impact']] if 'scores' in b and 'Impact' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Duration = [[b['scores']['Duration']] if 'scores' in b and 'Duration' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Entity_Density = [[b['scores']['Entity_Density']] if 'scores' in b and 'Entity_Density' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Market_Scope = [[b['scores']['Market_Scope']] if 'scores' in b and 'Market_Scope' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Time_Proximity = [[b['scores']['Time_Proximity']] if 'scores' in b and 'Time_Proximity' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Headline_Structure = [[b['scores']['Headline_Structure']] if 'scores' in b and 'Headline_Structure' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            Source_Recency = [[b['scores']['Source_Recency']] if 'scores' in b and 'Source_Recency' in b['scores'] else [0.0] for b in all_news[comp][self.all_date[date_ind]][0]]
            policy_related = [b['features']['policy_related'] if 'features' in b and 'policy_related' in b['features'] else "false" for b in all_news[comp][self.all_date[date_ind]][0]]
            investement_strategy = [b['features']['investement_strategy'] if 'features' in b and 'investement_strategy' in b['features'] else "false" for b in all_news[comp][self.all_date[date_ind]][0]]
            causal_factor = [b['features']['causal_factor'] if 'features' in b and 'causal_factor' in b['features'] else "false" for b in all_news[comp][self.all_date[date_ind]][0]]
            causal_impact = [b['features']['causal_impact'] if 'features' in b and 'causal_impact' in b['features'] else "neutral" for b in all_news[comp][self.all_date[date_ind]][0]]
            affected_by_time_series = [b['features']['affected_by_time_series'] if 'features' in b and 'affected_by_time_series' in b['features'] else "false" for b in all_news[comp][self.all_date[date_ind]][0]]
            ts_effect_direction = [b['features']['ts_effect_direction'] if 'features' in b and 'ts_effect_direction' in b['features'] else "neutral" for b in all_news[comp][self.all_date[date_ind]][0]]
            return (keywords,sectors,[Correlation, Sentiment, Importance, Impact, Duration, Entity_Density, Market_Scope, Time_Proximity, Headline_Structure, Source_Recency],policy_related,investement_strategy,causal_factor,causal_impact,affected_by_time_series,ts_effect_direction)
        return None

    def trace_back_price(self, comp, date, k = 5):
        if type(date) == int:
            temp_int = date
            date = self.all_date[date]
        date_obj = datetime.strptime(date, "%Y-%m-%d").weekday()
        ind = self.all_date.index(date)
        assert temp_int == ind
        if date_obj < 5:
            price_window_st = ind - (k-1 + (2 if datetime.strptime(self.all_date[(ind-(k-1))], "%Y-%m-%d").weekday()>date_obj else 0))
            price_window = list(range(price_window_st, ind+1))
        else:
            price_window_st = 0
            price_window = list(range(ind-(date_obj-4)-4, ind-(date_obj-4)+1))
        if price_window_st >= 0:
            if self.all_date_inverse[price_window[-1]] in self.all_price[comp]:
                return torch.cat([torch.tensor([self.all_price[comp][self.all_date_inverse[b]]]) for b in price_window if self.all_date_inverse[b] in self.all_price[comp]],dim=0), self.all_date_inverse[price_window[-1]] 
            else:
                return None, None
        else:
            return None, None
    
    def cal_label(self, comp, date):
        if type(date) == int:
            temp_int = date
            date = self.all_date[date]
        label_comp = self.all_label[comp]
        ind = self.all_date.index(date)
        assert temp_int == ind
        # +1: 6 2; 5 3; <=4 1
        if ind+1 == len(self.all_date):
            return None, None
        wd = datetime.strptime(self.all_date[ind], "%Y-%m-%d").weekday()
        temp = 1
        if wd == 4:
            temp = 3
        if ind+1 == len(self.all_date) or self.all_date[ind+temp ] not in label_comp:
            return None, temp
        else:
            return torch.tensor([1]) if float(label_comp[self.all_date[ind+ temp]]) > 0 else torch.tensor([0]), temp
    def cal_label1(self, comp, date_last, data_id):
        if date_last is None:
            return None, None
        
        label_comp = self.all_label[comp]
        
        # if type(date) == int:
        #     temp_int = date
        #     date = self.all_date[date]
        # label_comp = self.all_label[comp]
        # ind = self.all_date.index(date)
        # assert temp_int == ind
        ind= self.all_date.index(date_last)
        # print(date_last,data_id)
        wd = datetime.strptime(self.all_date[data_id], "%Y-%m-%d").weekday()
        temp = 1
        if wd == 4:
            temp = 3
        if ind+1 == len(self.all_date) or self.all_date[ind+temp ] not in label_comp:
            return None, temp
        else:
            return torch.tensor([1]) if float(label_comp[self.all_date[ind+ temp]]) > 0 else torch.tensor([0]), temp
    
    def _data_from_json_to_heterodata(self, data_id, fields, comp):
        if comp not in self.already:
            self.already[comp] = []
        hetero_data = HeteroData()
        current_start = {
            'score': 0,
            'other': 0,
            'keywords': 0,
            'news': 0,
            'virtual': 0,
            'sector': 0,
            'sc_avg': 0
        }
        padded_text = None
        news_by_date = self.get_news_token(comp, data_id)
        price, date_last = self.trace_back_price(comp, data_id)
        # label,temp = self.cal_label(comp, data_id)
        label, temp = self.cal_label1(comp, date_last, data_id)
        # print("299**",data_id,comp, self.all_date.index(self.all_date_inverse[data_id]), self.all_date_inverse[data_id],datetime.strptime(self.all_date_inverse[data_id], "%Y-%m-%d").weekday(),label,temp)
        
        # label = self.cal_label1(comp, data_id)
        if price is None or price.shape[0]!=5 or label is None:
            return None
        
        hetero_data['price'].x = price.unsqueeze(0)
        hetero_data['y'].x = label
        hetero_data['score'].x = torch.randn((0,10))
        hetero_data['other'].x = torch.randn((0,6,10)).long()
        hetero_data['keywords'].x = torch.randn((0,15)).long()
        hetero_data['news'].x = torch.randn((0,40)).long()
        hetero_data['virtual'].x = torch.randn((0,5)).long()
        hetero_data['sector'].x = torch.randn((0,10)).long()
        hetero_data['news_token'].x = torch.zeros((1,8,40)).long()
        hetero_data['sc_avg'].x = torch.zeros((1,10)).long()
        hetero_data['score_pad'].x = torch.zeros((1,8,10)).long()
            
        hetero_data['keywords','to','virtual'].edge_index = torch.randn((2,0)).long()
        hetero_data['keywords','to','virtual'].edge_attr = torch.randn((0,1))
        hetero_data['sector','to', 'score'].edge_index = torch.randn((2,0)).long()
        hetero_data['sector','to', 'score'].edge_attr = torch.randn((0,1))
        hetero_data['score','to', 'sector'].edge_index = torch.randn((2,0)).long()
        hetero_data['score','to', 'sector'].edge_attr = torch.randn((0,1))

        all_keys = ['score','other','news','virtual','sector','sc_avg']
        for key in all_keys:
            hetero_data[key,'to','price'].edge_index = torch.randn((2,0)).long()
            hetero_data[key,'to','price'].edge_attr = torch.randn((0,1))
        from itertools import chain

        
        if news_by_date is not None:
            if news_by_date is not None:
                padded_text = [self._tokenize_and_pad_text(new) for new in news_by_date]
                padded_text_pad = self._pad_texts_to_equal_length(padded_text, max_docs = 8,max_seq_len = 40)
            else:
                padded_text = [self._tokenize_and_pad_text("None")]
                padded_text_pad = self._pad_texts_to_equal_length([[]],max_docs = 8,max_seq_len = 40)


            
            other_feature_o = self.get_other_features(comp, data_id)
            keywords , sectors, sentiment_score, policy_related,investement_strategy,causal_factor,causal_impact,affected_by_time_series,ts_effect_direction = other_feature_o
            
            # print("321",sentiment_score,padded_text)
            # print("311**",sentiment_score)
            sentiment_score = [[b[0] for b in sublist] for sublist in sentiment_score]
            # score_avg = [[b[0] for b in sublist] for sublist in sentiment_score]
            
            score_avg = [[sum(b)/len(b) for b in sentiment_score]]
            # score_avg = 
            # print("317", sentiment_score)
            # print(score_avg)
            sentiment_score = list(map(list, zip(*sentiment_score)))
            scores_pad = self._pad_texts_to_equal_length(sentiment_score, max_docs = 8,max_seq_len = 10)
            virtuals = [["None"]*len(sentiment_score)]
            token_virtuals = [[self._tokenize_and_pad_text(new,5) for new in keyword] for keyword in virtuals]
            token_keywords = [[self._tokenize_and_pad_text(new,15) for new in keyword] for keyword in keywords]
            token_sectors = [[self._tokenize_and_pad_text(new,10) for new in keyword] for keyword in sectors]
            other_feature = [
            [self._tokenize_and_pad_text(b,10) for b in policy_related],
            [self._tokenize_and_pad_text(b,10) for b in investement_strategy],
            [self._tokenize_and_pad_text(b,10) for b in causal_factor],
            [self._tokenize_and_pad_text(b,10) for b in causal_impact],
            [self._tokenize_and_pad_text(b,10) for b in affected_by_time_series],
            [self._tokenize_and_pad_text(b,10) for b in ts_effect_direction]
            ]
            
            hetero_data['score'].x = torch.tensor(sentiment_score)
            hetero_data['other'].x = torch.tensor(other_feature).permute(1,0,2)
            hetero_data['keywords'].x = torch.tensor([token for tokens in token_keywords for token in tokens ])
            hetero_data['news'].x = torch.tensor(padded_text)
            hetero_data['virtual'].x = torch.tensor(token_virtuals).squeeze(0)
            hetero_data['sector'].x = torch.tensor([token for tokens in token_sectors for token in tokens])
            hetero_data['news_token'].x = torch.tensor(padded_text_pad).unsqueeze(0)
            hetero_data['sc_avg'].x = torch.tensor(score_avg)
            hetero_data['score_pad'].x = torch.tensor(scores_pad).unsqueeze(0)
            counter = [0, 0, 0, 0, 0, 0, 0, 0]
            all_other_feat = [[b[:1] for b in sentiment_score], [[6,6,6,6,6,6]]*len(sentiment_score), keywords, [[b] for b in news_by_date], [[0]]*len(sentiment_score), sectors, [[b[0] for b in score_avg]]*len(sentiment_score)]
            # print(all_other_feat[3],sentiment_score)
            # print("341**",all_other_feat)
            # print("342**",sectors)
            # print(all_other_feat)
            # print(len(keywords))
            rela_all = [
                (counter[0], counter[1], counter[2], counter[3], counter[4], counter[5], counter[6])
                for b in range(len(keywords))
                if (
                    (counter.__setitem__(0, counter[0] + len(all_other_feat[0][b]))) or
                    (counter.__setitem__(1, counter[1] + len(all_other_feat[1][b]))) or
                    (counter.__setitem__(2, counter[2] + len(all_other_feat[2][b]))) or
                    (counter.__setitem__(3, counter[3] + len(all_other_feat[3][b]))) or
                    (counter.__setitem__(4, counter[4] + len(all_other_feat[4][b]))) or
                    (counter.__setitem__(5, counter[5] + len(all_other_feat[5][b]))) or
                    (counter.__setitem__(6, counter[6] + len(all_other_feat[6][b]))) or
                    True
                )
            ]
            edges = {}
            attrs = {}
            last_temp_score = 0
            last_temp_sector = 0
            list_fields = ['score','other','news','virtual','sector','keywords','sc_avg']
            for rela_unit in rela_all:
                if sum(rela_unit)==0:
                    continue
                score_id, other_id, keywords_id, news_id, virtual_id, sector_id, sc_avg_id = rela_unit
                end_dict = {
                'score': score_id,
                'other': other_id,
                'news': news_id,
                'virtual': virtual_id,
                'sector': sector_id,
                'sc_avg': sc_avg_id, 
                }
                for key in end_dict.keys():
                    start = current_start[key]
                    end = end_dict[key]
                    if key == 'other':
                        end = (end+1)//6
                    sources = list(range(start, int(end)))
                    if key == 'score':
                        last_temp_score = start
                    if key == 'sector':
                        last_temp_sector = start
                    current_start[key] = int(end)
                    targets = [0] * len(sources)
                    if (key, 'to', 'price') not in edges:
                        edges[(key, 'to', 'price')] = []
                        attrs[(key, 'to', 'price')] = []
                    edges[(key, 'to', 'price')].extend(list(zip(sources, targets)))
                    attrs[(key, 'to', 'price')].extend([torch.tensor(list_fields.index(key))]*len(sources))

                start = current_start['keywords']
                end = keywords_id
                sources = list(range(start, int(end)))
                targets = [virtual_id-1] * len(sources)
                if ('keywords', 'to', 'virtual') not in edges:
                    edges[('keywords', 'to', 'virtual')] = []
                    attrs[('keywords', 'to', 'virtual')] = []

                edges[('keywords','to','virtual')].extend(list(zip(sources, targets)))
                attrs[('keywords','to','virtual')].extend([torch.tensor(list_fields.index('keywords'))]*len(sources))

                if ('sector','to','score') not in edges:
                    edges[('sector','to','score')] = []
                    attrs[('sector','to','score')] = []
                    
                    edges[('score','to','sector')] = []
                    attrs[('score','to','sector')] = []
                
                start = last_temp_sector
                end = sector_id
                sources = list(range(start, int(end)))
                start_2 = last_temp_score
                end_2 = score_id
                targets = [start_2] * len(sources)
                sources2 = list(range(start_2, int(end_2)))
                targets2 = [start] * len(sources2)
                edges[('sector','to','score')].extend(list(zip(sources, targets)))
                attrs[('sector','to','score')].extend([torch.tensor(len(list_fields))]*len(sources))
                edges[('score','to','sector')].extend(list(zip(targets, sources)))
                attrs[('score','to','sector')].extend([torch.tensor(torch.tensor(len(list_fields)+1))]*len(sources))
                current_start['keywords'] = int(keywords_id)
            del_key = []
            for edge_type in edges:
                if len(edges[edge_type]) == 0:
                    del_key.append(edge_type)
            edges = {key: value for key, value in edges.items() if key not in del_key}
            attrs = {key: value for key, value in attrs.items() if key not in del_key}

            edges = { k: torch.tensor(edges[k]).permute(1,0) for k in edges}
            attrs = { k: torch.tensor(attrs[k]).unsqueeze(1) for k in attrs}

            for key in current_start:
                if key != 'keywords':
                    if (key, 'to', 'price') in edges:
                        hetero_data[key,'to','price'].edge_index = edges[(key, 'to', 'price')]
                        hetero_data[key,'to','price'].edge_attr = attrs[(key, 'to', 'price')]
                        if key == 'sector':
                            hetero_data['sector','to', 'score'].edge_index = edges[('sector','to','score')]
                            hetero_data['sector','to', 'score'].edge_attr = attrs[('sector','to','score')]

                            hetero_data['score','to', 'sector'].edge_index = edges[('score', 'to', 'sector')]
                            hetero_data['score','to', 'sector'].edge_attr = attrs[('score', 'to', 'sector')]
                else:
                    if ('keywords','to','virtual') in edges:
                        hetero_data['keywords','to','virtual'].edge_index = edges[('keywords','to','virtual')]
                        hetero_data['keywords','to','virtual'].edge_attr = attrs[('keywords','to','virtual')]
        if self.metadata is None:
            self.metadata = hetero_data.metadata()
        hetero_data = ToUndirected()(hetero_data)
        return hetero_data


    def _data_to_heterodata(self, data, fields, comp):
        if comp not in self.already:
            self.already[comp] = []
        hetero_data = HeteroData()
        current_start = {
            'score': 0,
            'other': 0,
            'keywords': 0,
            'news': 0,
            'virtual': 0,
            'sector': 0
        }
        padded_text = None
        list_fields = list(current_start.keys())
        price = None
        for field in fields:
            if field == 'all_date_id':
                if int(data[field]) in self.already[comp]:
                    return None
                else:
                    self.already[comp].append(int(data[field]))
                price = self.trace_back_price(comp, self.all_date[int(data[field])])
                label = self.cal_label(comp,self.all_date[int(data[field])])
                if price is None or price.shape[0]!=5 or label is None:
                    return None
                news_by_date = self.get_news_token(comp, int(data[field]))
                if news_by_date is not None:
                    padded_text = [self._tokenize_and_pad_text(new) for new in news_by_date]
                    padded_text = self._pad_texts_to_equal_length(padded_text)
                else:
                    padded_text = self._pad_texts_to_equal_length([[]])
                other_features = self.get_other_features(comp, int(data[field]))
                keywords,sectors,sentiment_score,policy_related,investement_strategy,causal_factor,causal_impact,affected_by_time_series,ts_effect_direction = other_features
                padded_keywords = [[self._tokenize_and_pad_text(new,15) for new in keyword] for keyword in keywords]
                padded_keywords = [self._pad_texts_to_equal_length(key,) for key in padded_keywords]
                padded_sectors = [[self._tokenize_and_pad_text(new,10) for new in keyword] for keyword in sectors]
                padded_sectors = [self._pad_texts_to_equal_length(key) for key in padded_sectors]
                hetero_data['price'].x = price.unsqueeze(0)
                hetero_data['y'].x = label
        if hetero_data['price'].x.shape[-1] != 3:
            hetero_data['price'].x = price.unsqueeze(0)
            hetero_data['y'].x = label
        hetero_data['news_token'].x = torch.tensor(padded_text).unsqueeze(0)
        edges = {}
        attrs = {}
        last_temp_score = 0
        last_temp_sector = 0
        for rela_unit in data.rela:
            if torch.sum(rela_unit)==0:
                continue
            score_id, other_id, keywords_id, news_id, virtual_id, sector_id = rela_unit
            end_dict = {
            'score': score_id,
            'other': other_id,
            'news': news_id,
            'virtual': virtual_id,
            'sector': sector_id
            }
            for key in end_dict.keys():
                start = current_start[key]
                end = end_dict[key]
                if key == 'other':
                    end = (end+1)//6
                sources = list(range(start, int(end)))
                if key == 'score':
                    last_temp_score = start
                if key == 'sector':
                    last_temp_sector = start
                current_start[key] = int(end)
                targets = [0] * len(sources)
                if (key, 'to', 'price') not in edges:
                    edges[(key, 'to', 'price')] = []
                    attrs[(key, 'to', 'price')] = []
                edges[(key, 'to', 'price')].extend(list(zip(sources, targets)))
                attrs[(key, 'to', 'price')].extend([torch.tensor(list_fields.index(key))]*len(sources))

            start = current_start['keywords']
            end = keywords_id
            sources = list(range(start, int(end)))
            targets = [virtual_id-1] * len(sources)
            if ('keywords', 'to', 'virtual') not in edges:
                edges[('keywords', 'to', 'virtual')] = []
                attrs[('keywords', 'to', 'virtual')] = []

            edges[('keywords','to','virtual')].extend(list(zip(sources, targets)))
            attrs[('keywords','to','virtual')].extend([torch.tensor(list_fields.index('keywords'))]*len(sources))

            if ('sector','to','score') not in edges:
                edges[('sector','to','score')] = []
                attrs[('sector','to','score')] = []
                
                edges[('score','to','sector')] = []
                attrs[('score','to','sector')] = []
            
            start = last_temp_sector
            end = sector_id

            sources = list(range(start, int(end)))
            
            start_2 = last_temp_score
            end_2 = score_id

            targets = [start_2] * len(sources)
            sources2 = list(range(start_2, int(end_2)))
            targets2 = [start] * len(sources2)
            edges[('sector','to','score')].extend(list(zip(sources, targets)))
            attrs[('sector','to','score')].extend([torch.tensor(len(list_fields))]*len(sources))
            edges[('score','to','sector')].extend(list(zip(targets, sources)))
            attrs[('score','to','sector')].extend([torch.tensor(torch.tensor(len(list_fields)+1))]*len(sources))

        edges = { k: torch.tensor(edges[k]).permute(1,0) for k in edges}
        attrs = { k: torch.tensor(attrs[k]).unsqueeze(1) for k in attrs}

        for key in current_start:
            if key != 'keywords':
                hetero_data[key,'to','price'].edge_index = edges[(key, 'to', 'price')]
                hetero_data[key,'to','price'].edge_attr = attrs[(key, 'to', 'price')]
                if key == 'sector':
                    hetero_data['sector','to', 'score'].edge_index = edges[('sector','to','score')]
                    hetero_data['sector','to', 'score'].edge_attr = attrs[('sector','to','score')]

                    hetero_data['score','to', 'sector'].edge_index = edges[('score', 'to', 'sector')]
                    hetero_data['score','to', 'sector'].edge_attr = attrs[('score', 'to', 'sector')]
            else:
                hetero_data['keywords','to','virtual'].edge_index = edges[('keywords','to','virtual')]
                hetero_data['keywords','to','virtual'].edge_attr = attrs[('keywords','to','virtual')]

        return hetero_data
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.hetero_graphs_train[index]
        elif self.mode == 'val':
            return self.hetero_graphs_val[index]
        elif self.mode == 'test':
            return self.hetero_graphs_test[index]
        else:
            return self.hetero_graphs[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.hetero_graphs_train)
        elif self.mode == 'val':
            return len(self.hetero_graphs_val)
        elif self.mode == 'test':
            return len(self.hetero_graphs_test)
        else:
            return len(self.hetero_graphs)
    
    def get_train_set(self):
        return self.hetero_graphs_train
    
    def get_val_set(self):
        return self.hetero_graphs_val

    def get_test_set(self):
        return self.hetero_graphs_test

    def _build_hetero_graphs(self):
        hetero_graphs = []
        for data_cn in self.data_cn_list:
            for key in data_cn:
                self.comp_names.append(key)
                train_batches = []
                val_batches = []
                test_batches = []

                data_batch_all = data_cn[key]
                key_dict = list(data_batch_all[0][:][0].keys())
                k = 0
                for data_batch in data_batch_all:
                    k+=1
                    list_data = data_batch[:]     
                    train_batches = train_batches + [self._data_to_heterodata(b, key_dict, key) for b in list_data if b.all_date_id[0] <= self.period_id['tr_ed_id']]
                    val_batches = val_batches + [self._data_to_heterodata(b, key_dict,key) for b in list_data if (b.all_date_id[0] <= self.period_id['val_ed_id'] and b.all_date_id[-1] >= self.period_id['val_st_id'])]
                    test_batches = test_batches + [self._data_to_heterodata(b, key_dict,key) for b in list_data if (b.all_date_id[0] <= self.period_id['te_ed_id'] and b.all_date_id[-1] >= self.period_id['te_st_id'])]
                    
                train_batches = list(filter(lambda x: x != None, train_batches))
                val_batches = list(filter(lambda x: x != None, val_batches))
                test_batches = list(filter(lambda x: x != None, test_batches))
                if len(train_batches)!=0 and len(val_batches) !=0 and len(test_batches)!=0:
                    hetero_graphs.append([Batch.from_data_list(train_batches),Batch.from_data_list(val_batches),Batch.from_data_list(test_batches)])
                if len(train_batches)!=0:
                    self.hetero_graphs_train.append(Batch.from_data_list(train_batches))
                if len(val_batches)!=0:
                    self.hetero_graphs_val.append(Batch.from_data_list(val_batches))
                if len(test_batches)!=0:
                    self.hetero_graphs_test.append(Batch.from_data_list(test_batches))
        
        return hetero_graphs
    def _build_hetero_graphs_from_json(self):
        hetero_graphs = []
        key_dict = ['score','other','sector','keywords','news','virtual','price','y','all_date_id','rela']
        cp_N = 0
        last_train = []
        last_val = []
        last_test = []
        for comp in self.all_news_data:
            if not comp.startswith("."):
                cp_N += 1
                self.mode = 'all'
                self.comp_names.append(comp)
                train_batches = []
                val_batches = []
                test_batches = []
                k = 0
                # print("719**",self.all_label[comp])
                for data_batch in self.all_date:
                    k+=1
                    train_batches = train_batches + ([self._data_from_json_to_heterodata(self.all_date.index(data_batch), key_dict, comp)] if self.all_date.index(data_batch) <= self.period_id['tr_ed_id'] else [])
                    val_batches = val_batches + ([self._data_from_json_to_heterodata(self.all_date.index(data_batch), key_dict,comp) ] if (self.all_date.index(data_batch) <= self.period_id['val_ed_id'] and self.all_date.index(data_batch) >= self.period_id['val_st_id']) else [])
                    test_batches = test_batches + ([self._data_from_json_to_heterodata(self.all_date.index(data_batch), key_dict,comp) ] if (self.all_date.index(data_batch) <= self.period_id['te_ed_id'] and self.all_date.index(data_batch) >= self.period_id['te_st_id'])  else [])
                train_batches = list(filter(lambda x: x != None, train_batches))
                val_batches = list(filter(lambda x: x != None, val_batches))
                test_batches = list(filter(lambda x: x != None, test_batches))
                # print(train_batches[0])
                # print(train_batches[0]['score','to','price'].edge_index)
                if len(train_batches)!=0 and len(val_batches) !=0 and len(test_batches)!=0:
                    hetero_graphs.append([Batch.from_data_list(train_batches),Batch.from_data_list(val_batches),Batch.from_data_list(test_batches)])
                    # print(Batch.from_data_list(train_batches),"672**")
                if len(train_batches)!=0 :
                    temp = Batch.from_data_list(train_batches)
                    if temp['news'].x.shape[0] == 0:
                        if len(last_train)!=0:
                            # print("676**",Batch.from_data_list( last_train + train_batches ))
                            temp = Batch.from_data_list( last_train + train_batches )
                        else:
                            temp = None
                    if temp is not None:
                        self.hetero_graphs_train.append(temp)
                if len(val_batches)!=0:
                    temp = Batch.from_data_list(val_batches)
                    if temp['news'].x.shape[0] == 0:
                        if len(last_val)!=0:
                            temp = Batch.from_data_list( last_val + val_batches )
                        else:
                            temp = None
                    if temp is not None:
                        self.hetero_graphs_val.append(temp)
                if len(test_batches)!=0:
                    temp = Batch.from_data_list(test_batches)
                    if temp['news'].x.shape[0] == 0:
                        if len(last_test)!=0:
                            temp = Batch.from_data_list( last_test + test_batches )
                        else:
                            temp = None
                    if temp is not None:
                        self.hetero_graphs_test.append(temp)
                last_train = copy.copy(train_batches)
                # break
        with gzip.open("hetero_data_us_train.pkl.gz", "wb") as f:
            pickle.dump(self.hetero_graphs_train, f)
        with gzip.open("hetero_data_us_val.pkl.gz", "wb") as f:
            pickle.dump(self.hetero_graphs_val, f)
        with gzip.open("hetero_data_us_test.pkl.gz", "wb") as f:
            pickle.dump(self.hetero_graphs_test, f)
        return hetero_graphs
def edge_difference(edges1, edges2):
    """
    计算边集的差集 edges1 - edges2
    edges1: shape [2, N]
    edges2: shape [2, M]
    return: shape [2, K] 的边集差集
    """
    # 将边转换为元组的集合
    edges1_set = set(tuple(edge.tolist()) for edge in edges1.T)
    edges2_set = set(tuple(edge.tolist()) for edge in edges2.T)
    
    # 计算差集
    diff_set = edges1_set - edges2_set
    
    if len(diff_set) == 0:
        return torch.empty((2, 0), dtype=edges1.dtype)
    
    # 转换回tensor
    diff_edges = torch.tensor(list(diff_set)).T
    return diff_edges


if __name__ == '__main__':

    dataset = ProcessedDataset(data_path = "./",data_num = 1)
    epochs = args.epochs
    phase = [10]
    metadata = dataset.metadata
    metadata = (metadata[0],metadata[1]+[('price','rev_to','score'),('price','rev_to','sector'),('price','rev_to','virtual'),('price','rev_to','other'),('price','rev_to','news')])
    for i in range(5):
        model = NeuroDCG(metadata = metadata)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        device = torch.device("cuda:0")
        model.to(device)
        # pos_weight = torch.tensor([1]
        loss_fn = nn.BCEWithLogitsLoss(torch.tensor([1])).to(device)
    
        for epoch in range(epochs):
            dataset.mode = 'train'
            train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            i = 0
            model.train()
            predi_all = []
            predi_con_all = []
            true_all = []
            N_f = 5
            total_rank_all_train = np.array([0] * N_f)
            total_rank_all_val = np.array([0] * N_f)
            total_rank_all_test = np.array([0] * N_f)
            total_count_all_train = np.array([0] * N_f)
            total_count_all_val = np.array([0] * N_f)
            total_count_all_test = np.array([0] * N_f)
            dict1 = "./dict/dict_us.pkl"
            vocab = pkl.load(open(dict1,'rb'))
            inverse = {vocab[key]:key for key in vocab}
            
            for data in train_loader:
                # print(data)
                optimizer.zero_grad()
                # print(data)
                
                # print("773",data['news_token'].x,data['price'].x,data['y'].x, data['price'].x.shape,data['y'].x.shape)
                # for day in data['news_token'].x:
                #     for news in day:
                #         # print(news)
                #         test = ""
                #         for kw in news:
                #             test += inverse[int(kw)] +" "
                #         print(test)
                output, total_rank, ranks, out_cls, path, map_path, simu_all, grad_diff_all, edge_index_news, selected_edge_index_news, edge_index_keywords, selected_edge_index_keywords = model(data.to(device),epoch=epoch)
                # output, _, _, out_cls, _, _, _, _ = model(data.to(device),epoch=epoch)
                predictions = (torch.sigmoid(output)> 0.5).int()
                predictions_con = (torch.sigmoid(out_cls)> 0.5).int()
                total_rank_all_train += np.array([b.detach().cpu().numpy() for b in total_rank])
                total_count_all_train += np.array([b.shape[0] for b in ranks])
    
                predi_all.append(predictions)
                predi_con_all.append(predictions_con)
                true_labels = data['y'].x
                true_all.append(true_labels)
                mcc = calculate_mcc(predictions.detach().cpu().numpy(),true_labels.detach().cpu().numpy(),mode='a')
                train_acc = accuracy_score(true_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())
    
                mcc_con = calculate_mcc(predictions_con.detach().cpu().numpy(),true_labels.detach().cpu().numpy())
                train_acc_con = accuracy_score(true_labels.detach().cpu().numpy(), predictions_con.detach().cpu().numpy())
                loss1 = loss_fn(output, true_labels.float().unsqueeze(-1))
                loss2 = loss_fn(out_cls, true_labels.float().unsqueeze(-1))
                # 
                loss = loss2 + F.mse_loss(simu_all, grad_diff_all)
                loss.backward()
                i += 1
                optimizer.step()
            print(np.array(total_rank_all_train)/np.array(total_count_all_train))
            predi_all = torch.cat(predi_all,dim=0)
            predi_con_all = torch.cat(predi_con_all,dim=0)
            true_all = torch.cat(true_all,dim=0)
            mcc = calculate_mcc(predi_all.detach().cpu().numpy(),true_all.detach().cpu().numpy(),mode='c')
            mcc_con = calculate_mcc(predi_con_all.detach().cpu().numpy(),true_all.detach().cpu().numpy())
            train_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_all.detach().cpu().numpy())
            train_con_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_con_all.detach().cpu().numpy())
            print("train**",i, mcc,  train_acc, mcc_con, train_con_acc)
            model.eval()
        
            with torch.no_grad():
                i = 0
                predi_all = []
                predi_con_all = []
                true_all = []
                dataset.mode = 'val'
                val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                for data in val_loader:
                    # output, _, _, out_cls, _, _, _, _ = model(data.to(device),epoch='e')
                    output, total_rank, ranks, out_cls, path, map_path, simu_all, grad_diff_all, edge_index_news, selected_edge_index_news, edge_index_keywords, selected_edge_index_keywords = model(data.to(device),epoch = 'e')
                    total_rank_all_val += np.array([b.detach().cpu().numpy() for b in total_rank])
                    total_count_all_val += np.array([b.shape[0] for b in ranks])
                    predictions = (torch.sigmoid(output)> 0.5).int()
                    predictions_con = (torch.sigmoid(out_cls)> 0.5).int()
                    predi_all.append(predictions)
                    predi_con_all.append(predictions_con)
                    true_labels = data['y'].x
                    true_all.append(true_labels)
                    mcc = calculate_mcc(predictions.detach().cpu().numpy(),true_labels.detach().cpu().numpy())
                    train_acc = accuracy_score(true_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())
                    mcc_con = calculate_mcc(predictions_con.detach().cpu().numpy(),true_labels.detach().cpu().numpy())
                    train_acc_con = accuracy_score(true_labels.detach().cpu().numpy(), predictions_con.detach().cpu().numpy())
                    i += 1
                predi_all = torch.cat(predi_all,dim=0)
                predi_con_all = torch.cat(predi_con_all,dim=0)
                true_all = torch.cat(true_all,dim=0)
                mcc = calculate_mcc(predi_all.detach().cpu().numpy(),true_all.detach().cpu().numpy())
                mcc_con = calculate_mcc(predi_con_all.detach().cpu().numpy(),true_all.detach().cpu().numpy())
                train_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_all.detach().cpu().numpy())
                train_con_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_con_all.detach().cpu().numpy())
                print(np.array(total_rank_all_val)/np.array(total_count_all_val))
                print("val**",i, mcc, train_acc, mcc_con, train_con_acc)
            
                dataset.mode = 'test'
                test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                i = 0
                predi_all = []
                predi_con_all = []
                true_all = []
                for data in test_loader:
                    # output, _, _, out_cls, _, _, _, _ = model(data.to(device),epoch='e')
                    output, total_rank, ranks, out_cls, path, map_path, simu_all, grad_diff_all, edge_index_news, selected_edge_index_news, edge_index_keywords, selected_edge_index_keywords = model(data.to(device),epoch = 'e')
                    # e_d = edge_difference(edge_index_news, selected_edge_index_news)
                    # test = ""
                    # for e in e_d.T:
                    #     # print(e)
                    #     news = (data['news'].x)[e[0]]
                    #     # print(news)
                    #     for kw in news:
                    #         test += inverse[int(kw)] +" "
                    #     test += "\n"
                    #     test = test +str(e)+"\n"
                    # test1 = ""
                    # for e in selected_edge_index_news.T:
                    #     # print(e)
                    #     news = (data['news'].x)[e[0]]
                    #     for kw in news:
                    #         test1 += inverse[int(kw)] +" "
                    #     test1 += "\n"
                    #     test1 = test1+ str(e)+"\n"
                    # with open(str(epoch)+"test_news_del.txt","w") as f:
                    #     f.write(test)
                    #     f.close()
                    # with open(str(epoch)+"test_nes_rest.txt","w")as f:
                    #     f.write(test1)
                    #     f.close()
    
                    # e_d = edge_difference(edge_index_keywords, selected_edge_index_keywords)
                    # test = ""
                    # for e in e_d.T:
                    #     # print(e)
                    #     news = (data['keywords'].x)[e[0]]
                    #     # print(news)
                    #     for kw in news:
                    #         test += inverse[int(kw)] +" "
                    #     test += "\n"
                    #     test = test+str(e)+"\n"
                    # test1 = ""
                    # for e in selected_edge_index_news.T:
                    #     # print(e)
                    #     news = (data['keywords'].x)[e[0]]
                    #     for kw in news:
                    #         test1 += inverse[int(kw)] +" "
                    #     test1 += "\n"
                    #     test1 =test1+str(e)+"\n"
                    # with open(str(epoch)+"test_keywords_del.txt","w") as f:
                    #     f.write(test)
                    #     f.close()
                    # with open(str(epoch)+"test_keywords_rest.txt","w")as f:
                    #     f.write(test1)
                    #     f.close()
                    total_rank_all_test += np.array([b.detach().cpu().numpy() for b in total_rank])
                    total_count_all_test += np.array([b.shape[0] for b in ranks])
                    predictions = (torch.sigmoid(output)> 0.5).int()
                    predictions_con = (torch.sigmoid(out_cls)> 0.5).int()
                    predi_all.append(predictions)
                    predi_con_all.append(predictions_con)
                    true_labels = data['y'].x
                    true_all.append(true_labels)
                    mcc = calculate_mcc(predictions.detach().cpu().numpy(),true_labels.detach().cpu().numpy())
                    train_acc = accuracy_score(true_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())
    
                    mcc_con = calculate_mcc(predictions_con.detach().cpu().numpy(),true_labels.detach().cpu().numpy())
                    train_acc_con = accuracy_score(true_labels.detach().cpu().numpy(), predictions_con.detach().cpu().numpy())
                    i += 1
                # print(path, map_path)
                predi_all = torch.cat(predi_all,dim=0)
                predi_con_all = torch.cat(predi_con_all,dim=0)
                true_all = torch.cat(true_all,dim=0)
                mcc = calculate_mcc(predi_all.detach().cpu().numpy(),true_all.detach().cpu().numpy())
                mcc_con = calculate_mcc(predi_con_all.detach().cpu().numpy(),true_all.detach().cpu().numpy())
                train_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_all.detach().cpu().numpy())
                train_con_acc = accuracy_score(true_all.detach().cpu().numpy(), predi_con_all.detach().cpu().numpy())
                print(np.array(total_rank_all_test)/np.array(total_count_all_test))
                print("test**",i, mcc, train_acc, mcc_con, train_con_acc)