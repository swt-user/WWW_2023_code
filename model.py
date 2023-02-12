from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_user, num_item, dims, **kwargs):
        """
            loss_mode : 0 for pair-wise loss, 1 for softmax loss
        """
        super(BaseModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dims = dims
        self.count = 0
        

    def forward(self, user_id, pos_id, neg_id):
        pass
    

    def loss_function(self, neg_rat, neg_prob, pos_rat, user_id=None, reduction=False, weighted=False, pos_rank=None, lambda_w=1.0, loss_type=0, beta=1, **kwargs):
        
        if loss_type == 1:  # DNS loss
            neg_rat, _ = torch.topk(neg_rat, int(lambda_w), dim=-1)
            pred = torch.subtract(pos_rat.unsqueeze(1), neg_rat)
            loss = torch.negative(F.logsigmoid(pred))
            return torch.sum(loss, dim=-1).sum(-1)
            
        pred = torch.subtract(pos_rat.unsqueeze(1), neg_rat)
        
        with torch.no_grad():
            if weighted:
                if loss_type == 0:
                    importance = F.softmax(torch.negative(pred)/lambda_w - neg_prob, dim=1)
                    
                elif loss_type == 2:
                    Z = torch.negative(F.logsigmoid(pred))
                    tau = torch.sqrt( torch.var(Z, dim=1, keepdim=True)/(2*lambda_w) )
                    importance = F.softmax(Z/tau - neg_prob, dim=1) 
                 
                    
            else:
                importance = F.softmax(torch.ones_like(pred), dim=1)

        if pos_rank is not None:
            importance = importance * pos_rank

        weight_loss = torch.multiply(importance.detach(), torch.negative(F.logsigmoid(pred)))
        
        if reduction:
            return torch.sum(weight_loss, dim=-1).mean(-1)
        else:
            return torch.sum(weight_loss, dim=-1).sum(-1)

    def inference(self, user_id, item_id):
        pass

    def est_rank(self, user_id, pos_rat, candidate_items, sample_size):
        # (rank - 1)/N = (est_rank - 1)/sample_size
        candidate_rat = self.inference(user_id.repeat(candidate_items.shape[1],1).T, candidate_items)
        sorted_seq, _ = torch.sort(candidate_rat)
        quick_r = torch.searchsorted(sorted_seq, pos_rat.unsqueeze(-1))
        r = ((quick_r) * (self.num_item - 1 ) / sample_size ).floor().long()
        return self._rank_weight_pre[r]
    
    def cal_n(self,n):
        vec = 1 / torch.arange(0,n)
        return vec[1:].sum()
    


class BaseMF(BaseModel):
    def __init__(self, num_user, num_item, dims, pos_weight=False,**kwargs):
        super(BaseMF, self).__init__(num_user, num_item, dims)
        
        self._User_Embedding = nn.Embedding(self.num_user, self.dims) 
        self._Item_Embedding = nn.Embedding(self.num_item, self.dims)
        if pos_weight is True:
            self._rank_weight_pre = torch.tensor([self.cal_n(x) if x > 1 else 1 for x in range(self.num_item + 1)])
        self._init_emb()
    
    def _init_emb(self):
        nn.init.normal_(self._User_Embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self._Item_Embedding.weight, mean=0, std=0.1)
        
    
    def get_user_embs(self, eval_flag=True):
        return self._User_Embedding.weight
    
    def get_item_embs(self,  eval_flag=True):
        return self._Item_Embedding.weight
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        return (user_emb * item_emb).sum(-1)
    
    
    def forward(self, user_id, pos_id, neg_id):
        N = neg_id.shape[1]
        pos_rat = self.inference(user_id, pos_id)
        neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
        return pos_rat, neg_rat
    
    def predict(self, user_id):
        u_embeddings = self._User_Embedding(user_id)
        i_embeddings = self._Item_Embedding.weight # (self.num_item, dim)
        output = u_embeddings @ i_embeddings.transpose(-1, -2)
        eval_output = output.detach()
        return eval_output

    # def cal_rank(self, user_id, pos_rat):
    #     item_emb = self._Item_Embedding.weight
    #     user_emb = self._User_Embedding(user_id)
    #     all_rat = torch.matmul(user_emb, item_emb.T)
    #     sorted_seq, _ = torch.sort(all_rat)
    #     cal_r = torch.searchsorted(sorted_seq, pos_rat.unsqueeze(-1))
    #     return cal_r

class NCF(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
        self._W = nn.Linear(2 * dims, dims)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        gmf_out = user_emb * item_emb
        mlp_out = self._W(torch.cat([user_emb, item_emb], dim=-1))
        inferences = self._FC(torch.tanh(gmf_out + mlp_out))
        return inferences.squeeze(-1)
    
    # def forward(self, user_id, pos_id, neg_id):
    #     N = neg_id.shape[1]
    #     pos_rat = self.inference(user_id, pos_id)
    #     neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
    #     return pos_rat, neg_rat

class MLP(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
        self._W = nn.Linear(2 * dims, dims)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        mlp_out = self._W(torch.cat([user_emb, item_emb], dim=-1))
        inferences = self._FC(torch.tanh(mlp_out))
        return inferences.squeeze(-1)
    
    # def forward(self, user_id, pos_id, neg_id):
    #     N = neg_id.shape[1]
    #     pos_rat = self.inference(user_id, pos_id)
    #     neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
    #     return pos_rat, neg_rat

class GMF(BaseMF):
    def __init__(self, num_user, num_item, dims, pos_weight=False, **kwargs):
        super().__init__(num_user, num_item, dims, pos_weight=pos_weight, **kwargs)
        self._FC = nn.Linear(dims, 1, bias=False)
    
    def inference(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        gmf_out = user_emb * item_emb
        inferences = self._FC(gmf_out)
        return inferences.squeeze(-1)
    
    
class LightGCN(BaseModel):
    def __init__(self, num_user, num_item, dims, **kwargs):
        super().__init__(num_user, num_item, dims)
        
        self.Graph = self._convert_sp_mat_to_sp_tensor(kwargs['adj_mat']).to('cuda')
        self.n_layers = kwargs['n_layers']

        self._User_Embedding = nn.Embedding(self.num_user, self.dims) 
        self._Item_Embedding = nn.Embedding(self.num_item, self.dims)

        nn.init.xavier_normal_(self._User_Embedding.weight)
        nn.init.xavier_normal_(self._Item_Embedding.weight)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape).to('cuda')

    def compute(self):
        users_emb = self._User_Embedding.weight
        items_emb = self._Item_Embedding.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])

        return users, items

    def inference(self, user_id, item_id):
        all_users, all_items = self.compute()

        user_emb = all_users[user_id]
        item_emb = all_items[item_id]

        return (user_emb * item_emb).sum(-1)

    
    def forward(self, user_id, pos_id, neg_id):
        N = neg_id.shape[1]
        pos_rat = self.inference(user_id, pos_id)
        neg_rat = self.inference(user_id.unsqueeze(1).repeat(1,N), neg_id)
        return pos_rat, neg_rat
    
    def predict(self, user_id, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[user_id]
        items = torch.transpose(all_items[torch.tensor(items).to('cuda')], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()