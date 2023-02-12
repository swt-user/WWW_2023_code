import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class base_sampler(nn.Module):
    """
    Uniform sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(base_sampler, self).__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device
    
    def update_pool(self, model, **kwargs):
        pass
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.num_neg), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.num_neg, device=self.device))


class base_sampler_pop(base_sampler):
    """
    Popularity based sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(base_sampler_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)

        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
        pop_count = torch.squeeze(torch.from_numpy((mat.sum(axis=0).A).astype(np.float32)).to(device))
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / torch.sum(pop_count)
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        min_batch_size = min(4096, batch_size) # for memory limit
        cnt = batch_size // min_batch_size
        if (batch_size - cnt * min_batch_size) > 0:
            cnt += 1
        items = torch.zeros(batch_size, self.num_neg, dtype=torch.long, device=self.device)
        items_prob = torch.zeros(batch_size, self.num_neg, device=self.device)
        for c in range(cnt):
            end_index = min((c+1)*min_batch_size, batch_size)
            mmmm = end_index - c * min_batch_size
            items_min_batch = torch.multinomial(self.pop_prob.repeat(mmmm,1), self.num_neg)
            items_prob_min_batch = torch.log(self.pop_prob[items_min_batch])
            
            items[c * min_batch_size : end_index] = items_min_batch
            items_prob[c * min_batch_size : end_index] = items_prob_min_batch

        return items, items_prob


class two_pass(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.num_users = num_users
        self.sample_size = sample_size # importance sampling
        self.pool_size = pool_size # resample
        self.pool = torch.zeros(num_users, pool_size, device=device, dtype=torch.long)
    
    def update_pool(self, model, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
    
            neg_items, neg_q = self.sample_Q(user_batch)
            tmp_pool, tmp_score = self.re_sample(user_batch, model, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))
    
    def re_sample(self, user_batch, model, neg_items, log_neg_q):
        ratings = model.inference(user_batch.repeat(self.sample_size,1).T, neg_items)
        pred = ratings - log_neg_q
        sample_weight = F.softmax(pred, dim=-1)
        idices = torch.multinomial(sample_weight, self.pool_size, replacement=True)
        return torch.gather(neg_items, 1, idices), torch.gather(sample_weight, 1, idices)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            return
        
        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        self.pool[user_init] = tmp_pool[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
        candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
        self.pool[user_update] = torch.gather(candidate, 1, idx_k)
        return
    
    # @profile
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))

class two_pass_pop(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(two_pass_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)

        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
        pop_count = torch.squeeze(torch.from_numpy((mat.sum(axis=0).A).astype(np.float32)).to(device))
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / torch.sum(pop_count)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        items = torch.multinomial(self.pop_prob.repeat(batch_size,1), self.sample_size)
        return items, torch.log(self.pop_prob[items])

class two_pass_rank(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.candidate_items = torch.zeros(num_users, sample_size, device=self.device, dtype=torch.long)
    
    def update_pool(self, model, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
    
            neg_items, neg_q = self.sample_Q(user_batch)
            self.candidate_items[user_batch] = neg_items
            tmp_pool, tmp_score = self.re_sample(user_batch, model, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)
    


class two_pass_weight(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_weight, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag=False):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            self.pool_weight[user_batch] = tmp_score.detach()
            return

        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        if len(user_init) > 0:
            self.pool[user_init] = tmp_pool[idx]
            self.pool_weight[user_init] = tmp_score[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        if num_user_update > 0:
            idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
            candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
            candidate_weight = torch.cat([self.pool_weight[user_update], tmp_score[~idx]], dim=1)
            self.pool[user_update] = torch.gather(candidate, 1, idx_k)
            self.pool_weight[user_update] = torch.gather(candidate_weight, 1, idx_k).detach()
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        candidates_weight = self.pool_weight[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(torch.gather(candidates_weight, 1, idx_k))

class two_pass_weight_pop(two_pass_weight):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, mat, mode=0, **kwargs):
        super(two_pass_weight_pop, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
        pop_count = torch.squeeze(torch.from_numpy((mat.sum(axis=0).A).astype(np.float32)).to(device))
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / torch.sum(pop_count)
    
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        items = torch.multinomial(self.pop_prob.repeat(batch_size, 1), self.sample_size, replacement=True)
        return items, torch.log(self.pop_prob[items])

class two_pass_weight_rank(two_pass_weight):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.candidate_items = torch.zeros(num_users, sample_size, device=self.device, dtype=torch.long)
    
    def update_pool(self, model, batch_size=2048, cover_flag=False, **kwargs):
        num_batch = (self.num_users // batch_size) + 1
        for ii in range(num_batch):
            start_idx = ii * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            user_batch = torch.arange(start_idx, end_idx, device=self.device)
            
            neg_items, neg_q = self.sample_Q(user_batch)
            self.candidate_items[user_batch] = neg_items
            tmp_pool, tmp_score = self.re_sample(user_batch, model, neg_items, neg_q)
            self.__update_pool__(user_batch, tmp_pool, tmp_score, cover_flag=cover_flag)


class tapast(base_sampler):
    """
    The dynamic sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.pool_size = pool_size
        self.num_users = num_users

    def forward(self, user_id, model=None, **kwargs):
        batch_size = user_id.shape[0]
        pool = torch.randint(0, self.num_items, size=(batch_size, self.num_neg, self.pool_size), device=self.device)
        rats = model.inference(user_id.repeat(self.num_neg, 1).T, pool)
        r_v, r_idx = rats.max(dim=-1)
        return torch.gather(pool, 2, r_idx.unsqueeze(-1)).squeeze(-1), torch.exp(r_v)


class Tree_Kernel:
    def __init__(self, embs, device):
        # self.device = device
        embs = embs.detach().cpu().numpy()
        self.depth = math.ceil(math.log2(embs.shape[0]))
        self.tree = [None] * self.depth
        self.tree[0] = embs
        for d in range(1, self.depth):
            child = self.tree[d-1]
            if child.shape[0] % 2 != 0:
                # child = torch.cat([child, torch.zeros(1, child.shape[1], device=device)], dim=0)
                child = np.r_[child, np.zeros([1, child.shape[1]])]
            self.tree[d] = child[::2] + child[1::2]
    
    def sampling(self, query, neg):
        # rand_num_arr = torch.rand(neg, self.depth, device=self.device)
        # samples = torch.zeros(neg, dtype=torch.int32, device=self.device)
        # probs = torch.zeros(neg, device=self.device)
        query = query.detach().cpu().numpy()
        rand_num_arr = np.random.rand(neg, self.depth)
        samples = np.zeros(neg, dtype=np.int32)
        probs = np.zeros(neg, dtype=np.float32)
        for i in range(neg):
            selected = 0 
            rand_num = rand_num_arr[i]
            for d in range(self.depth, 0, -1):
                if self.tree[d-1].shape[0] > selected * 2 + 1:
                    # score = torch.matmul(self.tree[d-1][[selected * 2, selected * 2 + 1]], query)
                    score = np.matmul(self.tree[d-1][[selected * 2, selected * 2 + 1]], query)
                    idx_child = 0 if score[0]/(score[0]+score[1]) > rand_num[d-1] else 1
                    selected = selected * 2 + idx_child
                else:
                    selected = selected * 2
            probs[i] = score[idx_child]
            samples[i] = selected
        return samples, probs

class Adaptive_KernelBased(base_sampler):
    """
    The adaptive kernelbased sampler
    """
    @staticmethod
    def getkernel(embs, alpha, device):
        phi_t = torch.matmul(embs.unsqueeze(dim=2), embs.unsqueeze(dim=1))
        phi_ = torch.reshape(phi_t, (embs.shape[0], -1)) * math.sqrt(alpha)
        return torch.cat([phi_, torch.ones(embs.shape[0],1, device=device)], dim=1)


    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, alpha=1, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.alpha = alpha
        self.num_users = num_users
        self.pool_size = pool_size
    
    def update_pool(self, user_embs, item_embs, **kwargs):
        # to decrease the running time, we maintain the sampling pool
        self.kernel_item_embs = Adaptive_KernelBased.getkernel(item_embs, self.alpha, self.device)
        self.kernel_user_embs = Adaptive_KernelBased.getkernel(user_embs, self.alpha, self.device)
        self.tree = Tree_Kernel(self.kernel_item_embs, self.device)

        self.pool = torch.zeros(self.num_users, self.pool_size, device=self.device, dtype=torch.long)
        self.pool_weight = torch.zeros(self.num_users, self.pool_size, device=self.device)
        for idx in range(self.num_users):
            query = self.kernel_user_embs[idx]
            items, prob = self.tree.sampling(query, self.pool_size)
            self.pool[idx] = torch.tensor(items).to(self.device)
            self.pool_weight[idx] = torch.tensor(prob).to(self.device)

    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        sample_idx = torch.randint(0, self.pool_size, (batch_size, self.num_neg), device=self.device)
        return torch.gather(self.pool[user_id], 1, sample_idx), torch.gather(self.pool_weight[user_id], 1, sample_idx)
