import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.optim as optim
# from transformers import BertTokenizer
from datetime import datetime, timedelta
import pandas as pd


import os
import json
import torch
from torch.utils.data import Dataset
import pickle as pkl


class StockDataset(Dataset):
    def __init__(self, text_dir, sequence_dir, max_seq_len=40,mode='train'):
        """
        初始化数据集。
        Args:
            text_dir (str): 文本数据目录。
            sequence_file (str): 股票序列文件路径。
            vocab_file (str): 词汇表文件路径。
            max_seq_len (int): 文本序列的最大长度（用于截断或填充）。
        """
        self.text_dir = text_dir
        self.sequence_dir = sequence_dir
        # self.vocab = self.load_glove_embeddings()
        # self.vocab = pkl.load(open("/home/zhaokx/CMIN-Dataset-main/dict_tweet.pkl", 'rb'))
        self.vocab = pkl.load(open("dict_us.pkl", 'rb'))  # 加载词汇表
        self.max_seq_len = max_seq_len
        self.mode = mode
        # 加载文本数据
        self.text_data = self._load_text_data()
        # 加载股票序列数据
        self.stock_data = self._load_stock_data()
        self.labels = self.load_label()
        # 对齐数据
        # self.train_period = ['2018-01-01','2021-04-30']
        # self.dev_period = ['2021-09-01','2021-12-31']
        # self.train_period = ['2014-01-01','2014-12-31']
        # self.dev_period = ['2015-08-03','2015-09-10']
        # self.test_period = ['2015-10-01','2016-01-01']
        self.train_period = ['2018-01-01','2021-04-30']
        self.dev_period = ['2021-05-01','2021-08-31']
        self.test_period = ['2021-09-01','2021-12-31']
        self.data = self._align_data_by_time()

    def _load_text_data(self):
        """
        加载文本数据并按时间戳组织。
        """
        text_data = {}
        for comp in os.listdir(self.text_dir):
            comp_path = os.path.join(self.text_dir, comp)
            if os.path.isdir(comp_path):
                text_data[comp] ={}
                file_list = os.listdir(comp_path)
                file_list.sort()
                for file_name in file_list:
                    file_path = os.path.join(comp_path, file_name)
                    # print(file_path,"124**")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            time = entry["created_at"].split(" ")[0]
                            # time = entry["created_at"]
                            # parsed_time = datetime.strptime(time, "%a %b %d %H:%M:%S +0000 %Y")

                            # # # #转换为目标格式
                            # time = parsed_time.strftime("%Y-%m-%d")
                            text = entry["text"]
                            if time not in text_data[comp]:
                                text_data[comp][time] = []
                            text_data[comp][time].append(text)
                            # print(text)
        return text_data
    def load_glove_embeddings(self, file_path="/home/zhaokx/CMIN-Dataset-main/glove.twitter.27B.200d.txt", vocab_size=50000):
        vocab = {}
        with open(file_path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx >= vocab_size-1:
                    break
                word = line.split()[0]
                vocab[word] = idx  # word -> index
        # vocab["<unk>"] = str(len(vocab)-2)
        # vocab["<pad>"] = str(len(vocab)-1)
        return vocab
    def load_label(self):
        labels = {}
        for sequence_file in os.listdir(self.sequence_dir):
            path = os.path.join(self.sequence_dir,sequence_file)
            labels[sequence_file.replace(".txt","")] ={}
            # print(sequence_file)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    time = parts[0]
                    features = float(parts[1])
                    labels[sequence_file.replace(".txt","")][time] = features
            # break
        return labels

    def _load_stock_data(self):
        """
        加载股票序列数据。
        """
        stock_data = {}
        for sequence_file in os.listdir(self.sequence_dir):
            path = os.path.join(self.sequence_dir,sequence_file)
            stock_data[sequence_file.replace(".txt","")] ={}
            # print(sequence_file)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    time = parts[0]
                    features = [float(x) for x in parts[3:6]]
                    stock_data[sequence_file.replace(".txt","")][time] = features
            # break
        return stock_data

    def _align_data_by_time(self):
        """
        对齐文本数据和股票数据。
        """
        aligned_data_all = []
        # print(self.stock_data)
        for comp, times in self.stock_data.items():
            # print(comp)

            aligned_data = []
            # print(comp in self.stock_data)
            if comp in self.stock_data:
                for time in times:
                    # print(time,time in self.stock_data[comp])
                    if time in self.stock_data[comp]:
                        stock_features = self.stock_data[comp][time]
                        # print(stock_features,"180**")
                        if self.mode == 'train':
                            test = datetime.strptime(time, '%Y-%m-%d').date()
                            if datetime.strptime(self.train_period[0], '%Y-%m-%d').date()< test < datetime.strptime(self.train_period[1], '%Y-%m-%d').date():
                                aligned_data.append({
                                    "comp": comp,
                                    "time": time,
                                    "texts": self.text_data[comp][time] if time in self.text_data[comp] else [],
                                    "stock_features": stock_features,
                                    "label": self.labels[comp][time]
                                })
                        if self.mode == 'val':
                            test = datetime.strptime(time, '%Y-%m-%d').date()
                            if datetime.strptime(self.dev_period[0], '%Y-%m-%d').date()< test < datetime.strptime(self.dev_period[1], '%Y-%m-%d').date():
                                aligned_data.append({
                                    "comp": comp,
                                    "time": time,
                                    "texts": self.text_data[comp][time] if time in self.text_data[comp] else [],
                                    "stock_features": stock_features,
                                    "label": self.labels[comp][time]
                                })
                        if self.mode == 'test':
                            test = datetime.strptime(time, '%Y-%m-%d').date()
                            if datetime.strptime(self.test_period[0], '%Y-%m-%d').date()< test < datetime.strptime(self.test_period[1], '%Y-%m-%d').date():
                                aligned_data.append({
                                    "comp": comp,
                                    "time": time,
                                    "texts": self.text_data[comp][time] if time in self.text_data[comp] else [],
                                    "stock_features": stock_features,
                                    "label": self.labels[comp][time]
                                })
                        
                        
                aligned_data_all.append(aligned_data)
                # break
        # print(aligned_data)
        return aligned_data_all
    def _pad_texts_to_equal_length(self, tokenized_texts, max_docs=10):
        """
        对 tokenized_texts 填充或截断，使其数量达到 max_docs。
        Args:
            tokenized_texts (list of list): 当前样本中的所有文本的索引化序列。
            max_docs (int): 统一的文本序列数量。
        Returns:
            padded_texts (list of list): 填充后的 tokenized_texts。
        """
        empty_text = [self.vocab["<pad>"]] * self.max_seq_len # 单个空文本的填充值
        # 如果数量不足，填充空文本；如果超出，进行截断
        if len(tokenized_texts) < max_docs:
            tokenized_texts.extend([empty_text] * (max_docs - len(tokenized_texts)))
        else:
            tokenized_texts = tokenized_texts[:max_docs]
        return tokenized_texts

    def _tokenize_and_pad_text(self, text):
        """
        将文本索引化并填充到固定长度。
        """
        indices = [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        padded = indices[:self.max_seq_len] + [self.vocab["<pad>"]] * max(0, self.max_seq_len - len(indices))
        return padded

    def construct_edge_index(self,samples,mappings):
        """
        构造 edge_index，根据样本中时间戳的文本和股票数据。
        Args:
            samples (list of dict): 当前样本数据。
        Returns:
            edge_index (torch.LongTensor): 图的边索引 (2, num_edges)。
        """
        edges = []
        edges1 = []
        edges2 = []
        num_nodes = len(samples)  # 节点总数（时间步数量）
        # print("250",num_nodes)
        # 时间相关边：连接相邻时间戳的节点
        for i in range(num_nodes - 5):
            edges.append((i, i + 1,0))
            edges.append((i, i + 2,1))
            edges.append((i, i + 3,2))
            edges.append((i, i + 4,3))
            # edges.append((i, i + 5,4))
        for key in mappings:
            edges1.append((key,mappings[key],5))
        if len(mappings)!=0:
            keys = mappings.keys()
            min_key = min(keys)
            max_key = max(keys)
            for i in range(min_key,max_key):
                edges2.append((i,i+1,6))

        # 转换为 edge_index 格式
        if len(edges)!=0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_index2 = torch.tensor(edges2, dtype=torch.long).t()
            if len(edges1)!=0:
                edge_index1 = torch.tensor(edges1, dtype=torch.long).t()
                
                return edge_index,edge_index1,edge_index2
            return edge_index,None,edge_index2
        return None,None,None

    def __getitem__(self, idx):
        """
        获取单个样本，包括文本和股票特征。
        """
        samples = self.data[idx]
        all_text_tensors = []
        all_stock_tensors = []
        times = []
        labels = []
        mappings = {}
        comps = []
        text_id = len(samples)
        all_texts = []
        temp_id = text_id
        for i, sample in enumerate(samples):
            texts = sample["texts"]
            stock_features = sample["stock_features"]
            time = sample['time']
            # 对所有文本进行索引化和填充
            tokenized_texts = [self._tokenize_and_pad_text(text) for text in texts]
            # for k in range(len(tokenized_texts)):
            #     mappings[text_id] = i
            #     text_id += 1
            # temp_id = text_id
            
            # for text in texts:
            #     # print(torch.tensor(self._tokenize_and_pad_text(text),dtype=torch.long).shape,"321**")
            #     all_texts.append(torch.tensor(self._tokenize_and_pad_text(text),dtype=torch.long))  
            # close_today = samples[i]["stock_features"][2]
            # close_tomorrow =  samples[i+1]["stock_features"][2] if i+1<len(samples) else close_today
            label = 1 if (i+1 < len(samples) and samples[i+1]["label"] > 0) else 0
            # 将文本和股票特征转为张量
            tokenized_texts = self._pad_texts_to_equal_length(tokenized_texts)
            text_tensor = torch.tensor(tokenized_texts, dtype=torch.long)
            stock_tensor = torch.tensor(stock_features, dtype=torch.float)
            all_text_tensors.append(text_tensor)
            all_stock_tensors.append(stock_tensor)
            times.append(time)
            labels.append(torch.tensor(label))
            comps.append(sample["comp"])
            # print(text_tensor.shape)
        # edge_index,edge_index1,edge_index2 = self.construct_edge_index(samples,mappings)
        # print(torch.stack(all_text_tensors).shape,torch.stack(all_texts).shape,"338")
        return {
        "comp": comps,
        "time": times,  # 时间戳信息，列表形式
        "texts":  torch.stack(all_text_tensors) if all_text_tensors else None,  # 文本张量 (num_samples, num_docs, max_words)
        "stock_features": torch.stack(all_stock_tensors) if all_stock_tensors else None,  # 股票特征张量 (num_samples, num_docs, feature_dim)
        "label" : torch.stack(labels) if labels else None,
        # "edge_index": edge_index,
        # "edge_index1" : edge_index1,
        # "edge_index2" : edge_index2,
        # "all_text" : torch.stack(all_texts) if all_texts else None,
    }

    def __len__(self):
        """
        返回数据的长度。
        """
        return len(self.data)
if __name__ == "__main__":
    data = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/preprocessed")
    data[0]

