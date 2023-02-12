# this provides some basic utilities, such as matrix split, read file into a matrix

from operator import mod
import scipy as sp
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
from typing import List
import logging
import torch
import torch.nn as nn




def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)
    
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Evaluate(nn.Module):
    def __init__(self, device, topk=50, evaluate_mask=True, test_batch_size=1000):
        super(Evaluate, self).__init__()
        self.device = device
        self.evaluate_mask = evaluate_mask
        self.topk = topk
        self.test_batch_size= test_batch_size

    def change_topk(self, new_topk):
        self.topk = new_topk
    
    def test_GPU_model(self, model, train_mat, test_mat):
        '''
        Rec: [user_list, item_list] Tensor
        user_id: [id]
        train_mat: csr_matrix 
        test_mat: csr_matrix
        '''
        m = {}
        num_user, num_item = train_mat.shape
        num_batch = int(num_user / self.test_batch_size) + 1
        linear_idx = list(range(num_user))

        topk_binary_list = []
        # batch calculate
        for i in range(num_batch):
            user_id = linear_idx[i * self.test_batch_size: min(num_user, (i+1) * self.test_batch_size)]
            user_batch = torch.LongTensor(user_id).to(self.device)
            
            # calculate the score
            with torch.no_grad():
                scores = model.predict(user_batch)

            if self.evaluate_mask:
                scores[train_mat[user_id].nonzero()] = float('-inf')
            # topk finding
            values, indices = torch.topk(scores, self.topk, dim=-1, largest=True, sorted=True)
            # generate a binary matrix 𝑩 of size 𝑛 × 𝑚 to indicate the existence of an item in the test set 
            binary_matrix = torch.zeros_like(scores)
            binary_matrix[test_mat[user_id].nonzero()] = 1
            # use each row of matrix 𝑨 to index the same row in matrix 𝑩 and obtain a binary matrix 𝑪 of size 𝑛 × 𝐾
            topk_binary_batch = torch.gather(binary_matrix, 1, indices)
            topk_binary_list.append(topk_binary_batch)

        topk_binary = torch.cat(topk_binary_list, dim=0).cpu().numpy()

        # ignore the users which do not have pos items in test
        pos_len_list = np.array(test_mat.sum(axis=1)).squeeze()
        non_zero_indices = pos_len_list.nonzero()
        pos_len_list = pos_len_list[non_zero_indices]
        topk_binary = topk_binary[non_zero_indices]

        # calculate metric
        m['item_ndcg'] = self.ndcg_(topk_binary, pos_len_list).mean(axis=0)
        m['item_recall'] = self.recall_(topk_binary, pos_len_list).mean(axis=0)
        m['item_prec'] = self.precision_(topk_binary, pos_len_list).mean(axis=0)
        return m

    def recall_(self, pos_index, pos_len):
        r"""Recall_ (also known as sensitivity) is the fraction of the total amount of relevant instances
        that were actually retrieved

        .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

        .. math::
            \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}

        :math:`Rel_u` is the set of items relevant to user :math:`U`,
        :math:`Rec_u` is the top K items recommended to users.
        We obtain the result by calculating the average :math:`Recall@K` of each user.

        """
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


    def ndcg_(self, pos_index, pos_len):
        r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
        Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.

        .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

        .. math::
            \begin{gather}
                \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
                \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
                \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
                \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
            \end{gather}

        :math:`K` stands for recommending :math:`K` items.
        And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
        :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
        :math:`U^{te}` stands for all users in the test set.

        """
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, int(idx):] = idcg[row, int(idx) - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result

    def precision_(self, pos_index, pos_len):
        r"""Precision_ (also called positive predictive value) is the fraction of
        relevant instances among the retrieved instances

        .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

        .. math::
            \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

        :math:`Rel_u` is the set of items relevant to user :math:`U`,
        :math:`Rec_u` is the top K items recommended to users.
        We obtain the result by calculating the average :math:`Precision@K` of each user.

        """
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    
