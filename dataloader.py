from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import scipy.io as sci
import scipy as sp
import random
import numpy as np
import math
import os


class RecData(object):
    def __init__(self, dir, file_name):
        file_name = file_name + 'data.mat'
        self.file_name = os.path.join(dir, file_name)

    def get_data(self,ratio):
        mat = self.load_file(file_name=self.file_name)
        train_mat, test_mat = self.split_matrix(mat, ratio)
        # adj_mat
        adj_mat = self.build_sparse_graph(train_mat)
        return train_mat, test_mat, adj_mat
    
    def load_file(self,file_name=''):
        if file_name.endswith('.mat'):
            return sci.loadmat(file_name)['data']
        else:
            raise ValueError('not supported file type')

    def split_matrix(self, mat, ratio=0.8):
        mat = mat.tocsr()  #按行读取，即每一行为一个用户
        m,n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j]) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.sparse.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.sparse.csr_matrix((test_data, test_indices, test_indptr), (m,n))
        
        return train_mat, test_mat

    def build_sparse_graph(self, train_mat):
        # num_user is num_train_user
        # num_item is num_train_item
        def _bi_norm_lap(adj):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.sparse.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        num_user, num_item = train_mat.shape
        coo = train_mat.tocoo()
        cf = np.array([coo.row, coo.col])
        cf[:, 1] = cf[:, 1] + num_user  # [0, n_items) -> [n_users, n_users+n_items)
        cf_ = cf.copy()
        cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

        cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

        vals = [1.] * len(cf_)
        mat = sp.sparse.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(num_user+num_item, num_user+num_item))
        return _bi_norm_lap(mat)

# popular setting
class RecData2(object):
    def __init__(self, dir, file_name):
        file_name = file_name
        self.file_name = os.path.join(dir, file_name)

    def get_data(self,ratio):
        train_mat = sp.sparse.load_npz(os.path.join(self.file_name, 'train.npz'))
        test_mat = sp.sparse.load_npz(os.path.join(self.file_name, 'test.npz'))
        return train_mat, test_mat
    
        
        
class UserItemData(Dataset):
    def __init__(self, train_mat, train_flag=True):
        super(UserItemData, self).__init__()
        self.train = train_mat.tocoo()
    
    def __len__(self):
        # return self.train.shape[0]
        return self.train.nnz
    
    def __getitem__(self, idx):
        return self.train.row[idx].astype(np.int), self.train.col[idx].astype(np.int)
    
    def get_count(self):
        return np.bincount(self.train.row.astype(np.int), weights=None, minlength=0)
    
class data_prefetcher():
    def __init__(self, random_idx, train_data, bs, device):
        self.random_idx = random_idx
        self.train_data = train_data
        self.bs = bs 
        self.device = device
        self.stream = torch.cuda.Stream()
        self.count = 0
        self.max_count = int(len(train_data) / bs) + 1
        self.preload()

    def preload(self):
        if self.count < self.max_count:
            batch_idx = self.random_idx[(self.count * self.bs):min(len(self.train_data), (self.count + 1) * self.bs)]
            user_id, item_id = self.train_data.__getitem__(batch_idx)
            self.user_id, self.item_id = torch.LongTensor(user_id).to(self.device), torch.LongTensor(item_id).to(self.device)
            self.count += 1
        else:
            self.user_id = None
            self.item_id = None
            return
        with torch.cuda.stream(self.stream):
            self.user_id = self.user_id.cuda(non_blocking=True)
            self.item_id = self.item_id.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        user_id = self.user_id
        item_id = self.item_id
        self.preload()
        return user_id, item_id
