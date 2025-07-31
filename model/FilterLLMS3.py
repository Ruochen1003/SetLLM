import os
from typing import Mapping, Any, Union, Optional, Callable

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, LlamaPreTrainedModel
from torch.distributions import Normal

class std_mapper(nn.Module):
    """
    更稳定的 std 映射器：使用 log_std 建模，防止梯度爆炸
    输出范围控制在 [1e-3, 10] 之间
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        nn.init.constant_(self.fc2.weight, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)  # 初始 log_std = 0 → std = 1

    def forward(self, x):
        log_std = self.fc2(x)             # 输出在 [-1, 1]（经过 tanh 控制）
        std = F.softplus(log_std) + 1e-6       # std = e^log_std ∈ [0.37, 2.7]
        return std

class mean_mapper(nn.Module):
    """
    初始化一个输出为1的线性层
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class FilterLLM_S3(PreTrainedModel):
    def __init__(self, config, encoder, user_num):
        super().__init__(config)
        # Obtain the number of users, items, and vocabulary size
        self.cur_encoder = encoder
        self.cur_mean_mapper = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size)
        )
        #self.cur_std_mapper = std_mapper(config.hidden_size, config.hidden_size)
        self.cur_std = torch.zeros()
        self.user_embeddings = nn.Embedding(user_num, config.hidden_size)
        #self.lora_finetune(lora_config)

    # def log_prob_gaussian(self, sample, mean, std):
    #     var = std ** 2
    #     log_scale = torch.log(std)
    #     log_prob = -0.5 * (((sample - mean) ** 2) / var + 2 * log_scale + torch.log(
    #         torch.tensor(2 * torch.pi, device=sample.device)))
    #     return log_prob.sum(dim=-1)

    def reward_funtion(self, dis, ref_pro, ref_emb):
        logp_cur = dis.log_prob(ref_emb).sum(dim=-1)
        reward = logp_cur - ref_pro
        return reward

    def forward(self, input_ids_prompt, attention_mask, pos_emb, neg_emb, logp_ref_pos, logp_ref_neg, **kwargs):
        prompt_rep = self.encode_prompt(input_ids_prompt, attention_mask, **kwargs)
        cur_mean = self.cur_mean_mapper(prompt_rep)
        cur_std = self.cur_std_mapper(prompt_rep) #+ 1e-6  # 防止标准差为 0
        #cur_std = self.cur_std_mapper(prompt_rep)
        dist = Normal(loc=cur_mean, scale=cur_std)

        #cur_std = torch.clamp(self.cur_std_mapper(prompt_rep), min=1e-3)

        pos_reward = self.reward_funtion(dist, logp_ref_pos, pos_emb)
        neg_reward = self.reward_funtion(dist, logp_ref_neg, neg_emb)

        loss = torch.mean(torch.nn.functional.softplus(pos_reward - neg_reward)) #+ 0.01 * variance_reg - 0.1 * entropy_reg

        return {'loss': loss}

    def encode_prompt(self, input_ids_prompt, attention_mask,  **kwargs):
        outputs_main = self.cur_encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True, **kwargs)
        last_hidden_states = outputs_main.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[:, -1, :]
        return last_token_hidden_states

    def dpo_emb_sampling(self, input_ids, attention_mask, num_samples, **kwargs):
        last_token_hidden_states = self.encode_prompt(input_ids, attention_mask, **kwargs)
        mean = self.cur_mean_mapper(last_token_hidden_states)
        std = self.cur_std_mapper(last_token_hidden_states) #+ 1e-6  # 确保标准差不为零
        dist = Normal(loc=mean.unsqueeze(1), scale=std.unsqueeze(1))  # (B, 1, D) -> broadcasted

        sampled_embs = dist.rsample((num_samples,))  # (num_samples, B, D)
        log_probs = dist.log_prob(sampled_embs)  # (B, num_samples, D)
        log_probs = log_probs.sum(dim=-1).squeeze()  # (B, num_samples)

        sampled_embs = sampled_embs.squeeze()   # (B, num_samples, D)
        scores = torch.matmul(sampled_embs, self.user_embeddings.weight.T)
        _, sampled_actions = torch.topk(scores, k=20, dim=2)

        return sampled_actions, sampled_embs, log_probs

    def dpo_emb_sampling2(self, input_ids, attention_mask, num_samples, **kwargs):
        last_token_hidden_states = self.encode_prompt(input_ids, attention_mask, **kwargs)
        mean = self.cur_mean_mapper(last_token_hidden_states)  # (B, 1, D)
        std = self.cur_std_mapper(last_token_hidden_states)   # (B, 1, D)
        dist = Normal(loc=mean, scale=std)  # (B, 1, D)

        # ---- 拼接 mean + samples ----
        all_embs = torch.cat([
            mean.unsqueeze(dim=0),  # (B, 1, D) -> mean embedding
            dist.rsample((num_samples,)).squeeze(2)  # (num_samples, B, D)
        ], dim=0)
        all_embs = all_embs

        # ---- Log probs ----
        log_probs = dist.log_prob(all_embs).sum(dim=-1) # (num_samples+1, B)

        # ---- Scores & Actions ----
        scores = torch.matmul(all_embs, self.user_embeddings.weight.T)  # (num_samples+1, B, num_users)
        _, topk_actions = torch.topk(scores, k=20, dim=-1)  # (num_samples+1, B, 20)

        return topk_actions, all_embs, log_probs

    def dpo_emb_sampling3(self, input_ids, attention_mask, num_samples, radius = 0.01, **kwargs):
        last_token_hidden_states = self.encode_prompt(input_ids, attention_mask, **kwargs)
        mean = self.cur_mean_mapper(last_token_hidden_states)  # (B, 1, D)
        std = self.cur_std_mapper(last_token_hidden_states)   # (B, 1, D)
        dist = Normal(loc=mean, scale=std)  # (B, 1, D)

        # ---- 拼接 mean + samples ----
        all_embs = torch.cat([
            mean.unsqueeze(dim=0),  # (B, 1, D) -> mean embedding
            dist.rsample((num_samples,)).squeeze(2)  # (num_samples, B, D)
        ], dim=0)  # -> (num_samples+1, B, D)

        upper = dist.cdf(all_embs + radius)
        lower = dist.cdf(all_embs - radius)
        prob = (upper - lower).prod(dim=-1)
        log_probs = torch.log(prob + 1e-12)

        # ---- Scores & Actions ----
        scores = torch.matmul(all_embs, self.user_embeddings.weight.T)  # (num_samples+1, B, num_users)
        _, topk_actions = torch.topk(scores, k=20, dim=-1)  # (num_samples+1, B, 20)

        return topk_actions, all_embs, log_probs

    def init_model(self, path):
        # Load the state dict from the file
        self.cur_encoder = PeftModel.from_pretrained(self.cur_encoder, path)
        custom_weights = torch.load(os.path.join(path, "custom_weights.pt"))
        self.cur_mean_mapper.load_state_dict(custom_weights["mapper"])
        #self.cur_mean_mapper.load_state_dict(custom_weights["mean_mapper"])
        self.user_embeddings.load_state_dict(custom_weights["user_embeddings"])

    def init_model2(self, path):
        self.cur_encoder = PeftModel.from_pretrained(self.cur_encoder, path)
        custom_weights = torch.load(os.path.join(path, "custom_weights.pt"))
        self.cur_mean_mapper.load_state_dict(custom_weights["mean_mapper"])
        self.cur_std_mapper.load_state_dict(custom_weights["std_mapper"])
        self.user_embeddings.load_state_dict(custom_weights["user_embeddings"])





