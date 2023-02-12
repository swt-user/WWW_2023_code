import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sampler import *
from model import *
from dataloader import RecData, UserItemData
import argparse
import numpy as np
from utils import Evaluate
import utils as utils
import logging
import scipy as sp
import scipy.io
import datetime
import os
import json


def evaluate(model, train_mat, test_mat, config, logger, device, topk=50):
    logger.info("Start evaluation")
    model.eval()
    device = torch.device(config.device)
    with torch.no_grad():
        # users = torch.from_numpy(np.random.choice(user_num, min(user_num, 5000), False)).to(device)
        evals = Evaluate(config.device, topk=topk, evaluate_mask= config.evaluate_mask)
        m = evals.test_GPU_model(model, train_mat, test_mat)
    return m



# @profile
def train_model(model, sampler, train_mat, test_mat, config, logger):
    optimizer = utils_optim(config.learning_rate, config.weight_decay, config.momentum, model)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    device = torch.device(config.device)

    train_data = UserItemData(train_mat)
    user_pos_count = train_data.get_count()
    temp = 1/(1+np.exp(np.median(user_pos_count) - user_pos_count))
    user_ratio = torch.Tensor( config.rescale_hyp * temp).to(device)
    
    
    for epoch in range(config.epoch):
        sampler.zero_grad()
        if epoch % config.update_epoch < 1:
            sampler.update_pool(model, cover_flag=True)
        else:
            sampler.update_pool(model)

        model.train()
        sampler.train()

        loss_ = 0.0
        logger.info("Epoch %d"%epoch)
    
        num_batch = int(len(train_data) / config.batch_size) + 1
        random_idx = np.random.permutation(len(train_data))
        for i in range(num_batch):
            batch_idx = random_idx[(i * config.batch_size):min(len(train_data), (i + 1) * config.batch_size)]
            user_id, item_id = train_data.__getitem__(batch_idx)
            
            user_id, item_id = torch.LongTensor(user_id).to(device), torch.LongTensor(item_id).to(device)
            
            optimizer.zero_grad()
            
            
            neg_id, prob_neg = sampler(user_id, model=model)
            pos_rat, neg_rat = model(user_id, item_id, neg_id) 
            
            if sampler.__class__.__name__ in ["two_pass_rank", "two_pass_weight_rank"]:
                pos_rank = model.est_rank(user_id, pos_rat, sampler.candidate_items[user_id], config.sample_size)
                loss = model.loss_function(neg_rat, prob_neg, pos_rat, user_id=user_id, pos_rank=pos_rank.to(device), reduction=config.reduction, weighted=config.weighted, \
                    lambda_w = config.lambda_w, loss_type = config.loss_type, beta=config.beta, user_ratio=user_ratio)
            else:
                loss = model.loss_function(neg_rat, prob_neg, pos_rat, user_id=user_id, reduction=config.reduction, weighted=config.weighted, \
                    lambda_w = config.lambda_w, loss_type = config.loss_type, beta=config.beta, user_ratio=user_ratio)
            
            
            loss_ += loss
            loss.backward()
            optimizer.step()
            
        
        logger.info('--loss : %.2f '% (loss_))
        model.reinit()
        if epoch<20: # reinit
            moving_estimator = torch.zeros(len(train_data)).to('cuda')
            max_shift = torch.zeros(len(train_data)).to('cuda')
            
        scheduler.step()

        if (epoch % 10) == 0:
            result = evaluate(model, train_mat, test_mat, config, logger, device)
            logger.info('***************Eval_Res : NDCG@5,20,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][19], result['item_ndcg'][49]))
            logger.info('***************Eval_Res : RECALL@5,20,50 %.6f, %.6f, %.6f'%(result['item_recall'][4], result['item_recall'][19], result['item_recall'][49]))
        

def utils_optim(learning_rate, weight_decay, momentum, model):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError('Unkown optimizer!')
        

def main(config, logger=None):
    device = torch.device(config.device)
    data = RecData(config.data_dir, config.data)
    train_mat, test_mat, adj_mat = data.get_data(config.ratio)
    user_num, item_num = train_mat.shape
    config.num_user = user_num
    config.num_item = item_num
    logging.info('The shape of datasets: %d, %d'%(user_num, item_num))

    # sampler_list = [base_sampler, two_pass, two_pass_weight, base_sampler_pop, two_pass_pop, two_pass_weight_pop, tapast, two_pass_rank,two_pass_weight_rank ]
    global sampler_list
    assert config.sampler < len(sampler_list), ValueError("Not supported sampler")
    sampler = sampler_list[config.sampler](user_num, item_num, config.sample_size, config.pool_size, config.sample_num, device, mat=train_mat)


    # model_list = [BaseMF, NCF, GMF, MLP, LightGCN]
    global model_list
    assert config.model < len(model_list), ValueError("Not supported sampler")
    
    kwargs = {}
    if config.model == 4:
        kwargs['n_layers'] = config.lgn_n_layer
        kwargs['adj_mat'] = adj_mat
        
    if sampler.__class__.__name__ in ["two_pass_rank", "two_pass_weight_rank"]:
        model = model_list[config.model](user_num, item_num, config.dims, pos_weight=True, **kwargs)
    else:
        model = model_list[config.model](user_num, item_num, config.dims, **kwargs)


    model = model.to(device)
    sampler = sampler.to(device)
    train_model(model, sampler, train_mat, test_mat, config, logger)

    return evaluate(model, train_mat, test_mat, config, logger, device, topk=200)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('--data', default='ml100k', type=str, help='path of datafile')
    parser.add_argument('-d', '--dims', default=32, type=int, help='the dimenson of the latent vector for student model')
    parser.add_argument('-m', '--model', default=0, type=int, help='the model')
    parser.add_argument('-s','--sample_num', default=5, type=int, help='the number of sampled items')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=100, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=10, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='logs_test', type=str, help='the path for log files')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')
    parser.add_argument('--device', default='cuda', type=str, help='device for training, cuda or gpu')
    parser.add_argument('--sampler', default=0, type=int, help='the sampler, 0 : uniform, 1 : two pass, 2 : two pass with weight')
    parser.add_argument('--fix_seed', action='store_true', help='whether to fix the seed values')
    parser.add_argument('--step_size', default=5, type=int, help='step size for learning rate discount') 
    parser.add_argument('--gamma', default=0.95, type=float, help='discout for lr')
    parser.add_argument('--reduction', default=False, type=bool, help='loss if reduction')
    parser.add_argument('--sample_size', default=200, type=int, help='the number of samples for importance sampling')
    parser.add_argument('--pool_size', default=50, type=int)
    parser.add_argument('--update_epoch', default=10000, type=int, help='the intervals to update the sample pool')
    parser.add_argument('--weighted', action='store_true', help='whether weighted for the loss function')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay for the optimizer')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum for the optimizer')
    parser.add_argument('--anneal', default=0.001, type=float, help='the coefficient for the KL loss')
    parser.add_argument('--lambda_w', default=1, type=float, help='lambda for importance reweight')
    parser.add_argument('--beta', default=1, type=float, help='beta for SOPA')
    parser.add_argument('--rescale_hyp', default=0.1, type=float, help='rescale size for negative count')
    parser.add_argument('--loss_type', default=0, type=int, help='the type of loss')
    parser.add_argument('--log_info', default='0', type=str, help='the distinguish information for log')
    parser.add_argument('--evaluate_mask', default=True, type=bool, help='loss if reduction')
    parser.add_argument('--lgn_n_layer', default=3, type=int, help='lightgcn number of layers')

    model_list = [BaseMF, NCF, GMF, MLP, LightGCN]
    sampler_list = [base_sampler, two_pass, two_pass_weight, base_sampler_pop, two_pass_pop, two_pass_weight_pop, two_pass_rank, two_pass_weight_rank, tapast, Adaptive_KernelBased]
    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.log_info, config.data, model_list[config.model].__name__, sampler_list[config.sampler].__name__, str(config.sample_size), str(config.pool_size), str(config.sample_num), str(config.loss_type), timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = utils.get_logger(log_file_name)
    
    logger.info(config)
    logger.info([s.__name__ for s in sampler_list])
    logger.info([m.__name__ for m in model_list])
    
    if config.fix_seed:
        utils.setup_seed(config.seed)
    

    m = main(config, logger)
    # print('ndcg@5,10,50, ', m['item_ndcg'][[4,9,49]])

    logger.info('Eval_Res : NDCG@5,20,50 %.6f, %.6f, %.6f'%(m['item_ndcg'][4], m['item_ndcg'][19], m['item_ndcg'][49]))
    logger.info('Eval_Res : RECALL@5,20,50 %.6f, %.6f, %.6f'%(m['item_recall'][4], m['item_recall'][19], m['item_recall'][49]))
    logger.info('Eval_Res : Precision@5,20,50 %.6f, %.6f, %.6f'%(m['item_prec'][4], m['item_prec'][19], m['item_prec'][49]))
    logger.info("Finish")
    
    if not os.path.exists('log_summary'):
        os.makedirs('log_summary')
    with open('log_summary/'+ config.log_info + '.log', 'a') as f:
        config_dict = vars(config)
        config_dict['log_file_name'] = log_file_name
        config_dict['NDCG@5'] = m['item_ndcg'][4]
        config_dict['NDCG@20'] = m['item_ndcg'][19]
        config_dict['NDCG@50'] = m['item_ndcg'][49]
        config_dict['NDCG@100'] = m['item_ndcg'][99]
        config_dict['NDCG@150'] = m['item_ndcg'][149]
        config_dict['NDCG@200'] = m['item_ndcg'][199]
        config_dict['Recall@5'] = m['item_recall'][4]
        config_dict['Recall@20'] = m['item_recall'][19]
        config_dict['Recall@50'] = m['item_recall'][49]
        config_dict['Recall@100'] = m['item_recall'][99]
        config_dict['Recall@150'] = m['item_recall'][149]
        config_dict['Recall@200'] = m['item_recall'][199]
        config_dict['Precision@5'] = m['item_prec'][4]
        config_dict['Precision@20'] = m['item_prec'][19]
        config_dict['Precision@50'] = m['item_prec'][49]
        config_dict['Precision@100'] = m['item_prec'][99]
        config_dict['Precision@150'] = m['item_prec'][149]
        config_dict['Precision@200'] = m['item_prec'][199]
        f.write(json.dumps(config_dict))
        f.write("\n")
    