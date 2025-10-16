'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0]
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, mode='test'):
    warm_item, cold_item = dataset.warm_cold_item()
    if mode == 'test':
        cold_user_nb, warm_user_nb, overall_user_nb = dataset.test_user_nb()
        cold_user, warm_user, overall_user = dataset.test_user()
        exclude_cold, exclude_warm, exclude_overall = dataset.test_exclude()
        cold_res, _ = test(Recmodel, cold_user_nb, cold_user, exclude_cold, masked_items=warm_item)
        warm_res, _ = test(Recmodel, warm_user_nb, warm_user, exclude_warm, masked_items=cold_item)
        overall_res, _ = test(Recmodel, overall_user_nb, overall_user, exclude_overall, masked_items=None)
    elif mode == 'val':
        cold_user_nb, warm_user_nb, overall_user_nb = dataset.val_user_nb()
        cold_user, warm_user, overall_user = dataset.val_user()
        exclude_cold, exclude_warm, exclude_overall = dataset.val_exclude()
        cold_res, _ = test(Recmodel, cold_user_nb, cold_user, exclude_cold, masked_items=warm_item)
        warm_res, _ = test(Recmodel, warm_user_nb, warm_user, exclude_warm, masked_items=cold_item)
        overall_res, _ = test(Recmodel, overall_user_nb, overall_user, exclude_overall, masked_items=None)
    else:
        Exception("mode error")
    print("Cold Result:", cold_res)
    print("Warm Result:", warm_res)
    print("Overall Result:", overall_res)

    return cold_res, warm_res, overall_res


def test(Recmodel, ts_nei, ts_user, exclude_pair_cnt, masked_items=None):
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    max_K = max(world.topks)
    rating_list = []
    score_list = []
    groundTrue_list = []
    batch_size = world.config['test_u_batch_size']
    for i, beg in enumerate(range(0, len(ts_user), batch_size)):
        end = min(beg + batch_size, len(ts_user))
        batch_user = ts_user[beg:end]
        groundTrue = ts_nei[batch_user]
        batch_user = torch.Tensor(batch_user).long().to(world.device)
        rating_all_item = Recmodel.getUsersRating(batch_user)
        rating_all_item = rating_all_item.detach().cpu().numpy()

        # ================== exclude =======================
        exclude_pair = exclude_pair_cnt[0][exclude_pair_cnt[1][i]:exclude_pair_cnt[1][i + 1]]
        rating_all_item[exclude_pair[:, 0], exclude_pair[:, 1]] = -1e10

        if masked_items is not None:
            rating_all_item[:, masked_items] = -1e10
        # ===================================================

        top_scores, top_item_index = get_top_k(rating_all_item, max_K)

        score_list.append(top_scores)
        rating_list.append(top_item_index)
        groundTrue_list.append(groundTrue)

    X = zip(rating_list, groundTrue_list)
    pre_results = list(map(test_one_batch, X))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    n_ts_user = float(len(ts_user))
    results['recall'] /= n_ts_user
    results['precision'] /= n_ts_user
    results['ndcg'] /= n_ts_user
    return results, np.concatenate(score_list, axis=0)

def topk_numpy(arr, k, dim):
    idx = np.argpartition(-arr,kth=k,axis=dim)
    idx = idx.take(indices=range(k),axis=dim)
    val = np.take_along_axis(arr,indices=idx,axis=dim)
    sorted_idx = np.argsort(-val,axis=dim)
    idx = np.take_along_axis(idx,indices=sorted_idx,axis=dim)
    val = np.take_along_axis(val,indices=sorted_idx,axis=dim)
    return val,idx

def get_top_k(ratings, k):
    topk_val, topk_idx = topk_numpy(ratings, k, dim=-1)
    return topk_val, topk_idx
