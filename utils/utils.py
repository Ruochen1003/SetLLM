import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from Dataset.BPRDataset import BPRDataset
from Dataset.DPODataset import DPODataset
from Dataset.DPODataset_action import DPODataset_action
from Dataset.DPODataset_update import DPODataset_update
from Dataset.BPRDateset_update import BPRDataset_update

def test_per_item(Recmodel, item_ids, groundTrue, args, device):
    batch_size = args.test_batch
    item_ids = torch.tensor(item_ids).long().to(device)
    avg_scores = torch.zeros(len(item_ids), dtype=torch.float32).to(device)

    for i, beg in enumerate(range(0, len(item_ids), batch_size)):
        end = min(beg + batch_size, len(item_ids))
        batch_item = item_ids[beg:end]
        batch_item_tensor = torch.tensor(batch_item).long().to(device)
        rating_all_user = Recmodel.getItemsRating(batch_item_tensor)
        for j in range(end - beg):
            gt_users = groundTrue[j]
            if len(gt_users) > 0:
                scores = rating_all_user[j, gt_users]
                avg_scores[beg + j] = scores.mean()
            else:
                avg_scores[beg + j] = 0.0  # groundTruth 为空时，得分为 0

    return avg_scores

def test_per_item2(item_embs, user_emb_table, groundTrue, args, device):
    batch_size = args.test_batch
    avg_scores = torch.zeros(len(item_embs), dtype=torch.float32).to(device)
    all_user_emb = user_emb_table.weight
    for i, beg in enumerate(range(0, len(item_embs), batch_size)):
        end = min(beg + batch_size, len(item_embs))
        batch_item_emb = item_embs[beg:end]
        rating_all_user = torch.matmul(batch_item_emb, all_user_emb.t())
        for j in range(end - beg):
            gt_users = groundTrue[j]
            if len(gt_users) > 0:
                scores = rating_all_user[j, gt_users]
                avg_scores[beg + j] = scores.mean()
            else:
                avg_scores[beg + j] = 0.0  # groundTruth 为空时，得分为 0

    return avg_scores

def test_per_item4(item_embs, user_emb_table, groundTrue, args, device):
    batch_size = args.test_batch
    all_ndcgs = []
    all_avg_scores = []
    all_user_emb = user_emb_table.weight

    for beg in range(0, len(item_embs), batch_size):
        end = min(beg + batch_size, len(item_embs))
        batch_item_emb = item_embs[beg:end]
        rating_all_user = torch.matmul(batch_item_emb, all_user_emb.t())

        # ---- NDCG ----
        _, topk_users = torch.topk(rating_all_user, k=20, dim=1)  # (batch_size, 20)
        batch_ndcgs = test_one_batch_ndcg(topk_users, groundTrue[beg:end], k=20)
        all_ndcgs.append(batch_ndcgs.cpu())

        # ---- 平均点积 ----，直接利用 rating_all_user
        batch_avg_scores = []
        for i in range(end - beg):
            gt_users = list(groundTrue[beg + i])  # groundTruth 是 set 或 list
            if not gt_users:
                avg_score = torch.tensor(0.0, device=device)
            else:
                avg_score = rating_all_user[i, gt_users].mean()
            batch_avg_scores.append(avg_score)

        batch_avg_scores = torch.stack(batch_avg_scores)  # (batch_size,)
        all_avg_scores.append(batch_avg_scores.cpu())

    return torch.cat(all_ndcgs, dim=0), torch.cat(all_avg_scores, dim=0)

def test_per_item3(Recmodel, item_ids, groundTrue, args, device):
    batch_size = args.test_batch
    all_ndcgs = []
    all_avg_scores = []

    for beg in range(0, len(item_ids), batch_size):
        end = min(beg + batch_size, len(item_ids))
        batch_item = item_ids[beg:end]
        if isinstance(batch_item, list):
            batch_item_tensor = torch.tensor(batch_item).long().to(device)
        else:
            batch_item_tensor = batch_item.to(device)

        rating_all_user = Recmodel.getItemsRating(batch_item_tensor)  # (batch_size, num_users)

        # ---- NDCG ----
        _, topk_users = torch.topk(rating_all_user, k=20, dim=1)  # (batch_size, 20)
        batch_ndcgs = test_one_batch_ndcg(topk_users, groundTrue[beg:end], k=20)
        all_ndcgs.append(batch_ndcgs.cpu())

        # ---- 平均点积 ----，直接利用 rating_all_user
        batch_avg_scores = []
        for i in range(end - beg):
            gt_users = list(groundTrue[beg + i])  # groundTruth 是 set 或 list
            if not gt_users:
                avg_score = torch.tensor(0.0, device=device)
            else:
                avg_score = rating_all_user[i, gt_users].mean()
            batch_avg_scores.append(avg_score)

        batch_avg_scores = torch.stack(batch_avg_scores)  # (batch_size,)
        all_avg_scores.append(batch_avg_scores.cpu())

    return torch.cat(all_ndcgs, dim=0), torch.cat(all_avg_scores, dim=0)

def test_one_batch_ndcg(topk_users: torch.Tensor, groundTruth: list, k: int = 20) -> torch.Tensor:
    """
    纯 Tensor 版本，计算一个 batch 中每个 item 的 nDCG。

    参数:
        topk_users (torch.Tensor): shape (batch_size, k)，推荐的用户 id。
        groundTruth (list of lists or sets): 每个 item 的真实相关用户 id。
        k (int): 推荐列表长度，默认 20。

    返回:
        torch.Tensor: shape (batch_size,)，每个 item 的 nDCG 值。
    """
    batch_size = topk_users.shape[0]
    ndcgs = torch.zeros(batch_size, dtype=torch.float32)

    device = topk_users.device

    # 预计算 IDCG，最多为 k 位
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=device)
    idcg_table = torch.cumsum(1.0 / torch.log2(ranks + 1), dim=0)

    for i in range(batch_size):
        gt_set = set(groundTruth[i])
        if len(gt_set) == 0:
            continue
        pred = topk_users[i]  # (k,)

        # 计算 DCG：如果命中 → 1/log2(rank+2)，否则为 0
        hits = torch.tensor([1.0 / torch.log2(torch.tensor(rank + 2.0, device=device))
                             if user.item() in gt_set else 0.0
                             for rank, user in enumerate(pred)], device=device)
        dcg = hits.sum()

        idcg = idcg_table[min(len(gt_set), k) - 1]
        ndcgs[i] = dcg / idcg

    return ndcgs

def test_one_batch_hit(topk_users: torch.Tensor, groundTruth: list, k: int = 20) -> torch.Tensor:
    """
    纯 Tensor 版本，计算一个 batch 中每个 item 的 Hit@K。

    参数:
        topk_users (torch.Tensor): shape (batch_size, k)，推荐的用户 id。
        groundTruth (list of lists or sets): 每个 item 的真实相关用户 id。
        k (int): 推荐列表长度，默认 20。

    返回:
        torch.Tensor: shape (batch_size,)，每个 item 的 Hit 值 (0 或 1)。
    """
    batch_size = topk_users.shape[0]
    hits = torch.zeros(batch_size, dtype=torch.float32, device=topk_users.device)

    for i in range(batch_size):
        gt_set = set(groundTruth[i])
        if len(gt_set) == 0:
            continue
        pred = topk_users[i]
        # 判断是否至少有一个命中
        if any(user.item() in gt_set for user in pred):
            hits[i] = 1.0

    return hits

def dpo_sample(sample_model, dataloader, sample_num):
    item_ids_list = []
    sampled_action_list = []
    sampled_emb_list = []
    sampled_action_probs_list = []
    #user_emb_weight = user_embeddings.weight.to(sample_model.device)
    for batch in tqdm(dataloader):
        batch['input_ids_prompt'].to(sample_model.device)
        batch['attention_mask'].to(sample_model.device)
        sampled_actions, sampled_embs, log_probs = sample_model.dpo_emb_sampling2(batch['input_ids_prompt'], batch['attention_mask'], sample_num)
        item_ids_list.append(batch['item_id'])
        sampled_emb_list.append(sampled_embs)
        sampled_action_list.append(sampled_actions)
        sampled_action_probs_list.append(log_probs)

    return item_ids_list, sampled_action_list, sampled_emb_list, sampled_action_probs_list

def shuffle_users(sampled_users, sampled_emb_list, sampled_action_probs_list, require_perms=False):
    """
    对 sampled_users 的第0维(矩阵个数)交叉打乱，同时同步打乱 sampled_emb_list 和 sampled_action_probs_list
    """
    num_matrices, rows, cols = sampled_users.shape

    # 为每个行位置生成一个 num_matrices 长度的打乱索引 (rows, num_matrices)
    perms = torch.stack([torch.randperm(num_matrices) for _ in range(rows)])  # (rows, num_matrices)

    # 构建批次索引：行号广播成 (rows, num_matrices)
    row_indices = torch.arange(rows).unsqueeze(1).expand(-1, num_matrices)

    # 对 sampled_users 打乱，得到 (rows, num_matrices, cols)
    shuffled_users = sampled_users[perms, row_indices]  # (rows, num_matrices, cols)
    shuffled_users = shuffled_users.permute(1, 0, 2)    # (num_matrices, rows, cols)

    # 对 sampled_emb_list 打乱，得到 (rows, num_matrices, feature_dim)
    shuffled_emb = sampled_emb_list[perms, row_indices]  # (rows, num_matrices, 2048)
    shuffled_emb = shuffled_emb.permute(1, 0, 2)         # (num_matrices, rows, 2048)

    # 对 sampled_action_probs_list 打乱，得到 (rows, num_matrices)
    shuffled_probs = sampled_action_probs_list[perms, row_indices]  # (rows, num_matrices)
    shuffled_probs = shuffled_probs.permute(1, 0)
    # (num_matrices, rows)
    if require_perms:
        return shuffled_users, shuffled_emb, shuffled_probs, perms
    else:
        return shuffled_users, shuffled_emb, shuffled_probs

def generate_bpr_training_dataset(items, sampled_users, sampled_emb_list, sampled_action_probs_list, require_perms=False):
    dataset_list = []
    if require_perms:
        shuffled_users, shuffled_emb, shuffled_probs, perms = shuffle_users(sampled_users, sampled_emb_list, sampled_action_probs_list, require_perms)
        for train_data, emb, prob in zip(shuffled_users, shuffled_emb, shuffled_probs):
            dataset_list.append(BPRDataset(items, train_data, emb, prob))
        return dataset_list, perms
    else:
        #shuffled_users, shuffled_emb, shuffled_probs = shuffle_users(sampled_users, sampled_emb_list, sampled_action_probs_list, require_perms)
        shuffled_users, shuffled_emb, shuffled_probs = sampled_users, sampled_emb_list, sampled_action_probs_list
        for train_data, emb, prob in zip(shuffled_users, shuffled_emb, shuffled_probs):
            dataset_list.append(BPRDataset(items, train_data, emb, prob))
        return dataset_list

def generate_bpr_training_dataset_update(items, sampled_users, sampled_emb_list, sampled_action_probs_list, test_index, all_interacted_user_ids, ground_indices, require_perms=False):
    test_items = items[test_index]
    test_sampled_users = sampled_users[:,test_index, :]
    test_sampled_emb_list = sampled_emb_list[:,test_index, :]
    test_sampled_action_probs_list = sampled_action_probs_list[:,test_index]
    ground_items = items[ground_indices]
    all_interacted_user_ids = [all_interacted_user_ids[i] for i in ground_indices.tolist()]
    dataset_list = []
    if require_perms:
        shuffled_users, shuffled_emb, shuffled_probs, perms = shuffle_users(test_sampled_users, test_sampled_emb_list, test_sampled_action_probs_list, require_perms)
        for train_data, emb, prob in zip(shuffled_users, shuffled_emb, shuffled_probs):
            dataset_list.append(BPRDataset_update(test_items, train_data, emb, prob, ground_items, all_interacted_user_ids))
        return dataset_list, perms
    else:
        #shuffled_users, shuffled_emb, shuffled_probs = shuffle_users(test_sampled_users, test_sampled_emb_list, test_sampled_action_probs_list, require_perms)
        shuffled_users, shuffled_emb, shuffled_probs = test_sampled_users, test_sampled_emb_list, test_sampled_action_probs_list
        for train_data, emb, prob in zip(shuffled_users, shuffled_emb, shuffled_probs):
            dataset_list.append(BPRDataset_update(test_items, train_data, emb, prob, ground_items, all_interacted_user_ids))
        return dataset_list



def generate_bpr_training_dataset2(items, sampled_users, sampled_emb_list, sampled_action_probs_list):
    dataset_list = []
    #\shuffled_users, shuffled_emb, shuffled_probs, perms = shuffle_users(sampled_users, sampled_emb_list, sampled_action_probs_list, require_perms)
    for train_data, emb, prob in zip(sampled_users, sampled_emb_list, sampled_action_probs_list):
        dataset_list.append(BPRDataset(items, train_data, emb, prob))
    return dataset_list


def save_dpo_dataset(path, item_ids, score_list, embedding_list, prob_list, top_k):
    torch.save({
        'item_ids': item_ids,
        'score_list': score_list,
        'embedding_list': embedding_list,
        'prob_list': prob_list,
        'top_k': top_k
    }, path)


def build_dpo_embedding_dataset(item_ids, score_list, embedding_list, prob_list, item_content_path, dataset,
                                top_k=2, train_ratio=0.9,
                                save_train_path=None, save_test_path=None):
    """
    构建 DPO 数据集，并保存为 numpy 格式文件 (.npz)。

    参数:
        item_ids (torch.Tensor): shape=(N,)
        score_list (torch.Tensor): shape=(N,)
        embedding_list (torch.Tensor): shape=(N, D)
        prob_list (torch.Tensor): shape=(N,)
        top_k (int): top-k 配置
        train_ratio (float): 训练集比例
        save_train_path (str): 训练集保存路径 (如 'train_data.npz')
        save_test_path (str): 测试集保存路径 (如 'test_data.npz')

    返回:
        train_dataset, test_dataset
    """

    # 构建 Dataset
    full_dataset = DPODataset(item_ids, score_list, embedding_list, prob_list, item_content_path, dataset, top_k)
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    return train_dataset, test_dataset

def build_dpo_embedding_dataset_update(item_ids, score_list, embedding_list, prob_list, item_content_path, dataset,
                                top_k=2, train_ratio=0.9,
                                save_train_path=None, save_test_path=None):
    """
    构建 DPO 数据集，并保存为 numpy 格式文件 (.npz)。

    参数:
        item_ids (torch.Tensor): shape=(N,)
        score_list (torch.Tensor): shape=(N,)
        embedding_list (torch.Tensor): shape=(N, D)
        prob_list (torch.Tensor): shape=(N,)
        top_k (int): top-k 配置
        train_ratio (float): 训练集比例
        save_train_path (str): 训练集保存路径 (如 'train_data.npz')
        save_test_path (str): 测试集保存路径 (如 'test_data.npz')

    返回:
        train_dataset, test_dataset
    """

    # 构建 Dataset
    full_dataset = DPODataset_update(item_ids, score_list, embedding_list, prob_list, item_content_path, dataset, top_k)
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    return train_dataset, test_dataset


def build_dpo_embedding_dataset_action(item_ids, score_list, embedding_list, action_list, item_content_path, dataset,
                                top_k=2, train_ratio=0.9,
                                save_train_path=None, save_test_path=None):
    """
    构建 DPO 数据集，并保存为 numpy 格式文件 (.npz)。

    参数:
        item_ids (torch.Tensor): shape=(N,)
        score_list (torch.Tensor): shape=(N,)
        embedding_list (torch.Tensor): shape=(N, D)
        prob_list (torch.Tensor): shape=(N,)
        top_k (int): top-k 配置
        train_ratio (float): 训练集比例
        save_train_path (str): 训练集保存路径 (如 'train_data.npz')
        save_test_path (str): 测试集保存路径 (如 'test_data.npz')

    返回:
        train_dataset, test_dataset
    """

    # 构建 Dataset
    full_dataset = DPODataset_action(item_ids, score_list, action_list, item_content_path, dataset, top_k)
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    return train_dataset, test_dataset





