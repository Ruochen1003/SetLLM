import os
from typing import Mapping, Any, Union, Optional, Callable

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, LlamaPreTrainedModel
from torch.distributions import Normal

class FilterLLM_S3(PreTrainedModel):
    def __init__(self, config, encoder, user_num, fixed_std=0.5):
        super().__init__(config)
        self.cur_encoder = encoder
        self.fixed_std = fixed_std  # 设置固定 std

        # 映射用于生成 mean
        self.cur_mean_mapper = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.user_embeddings = nn.Embedding(user_num, config.hidden_size)
        self.margin = 1

    def reward_funtion(self, dist, ref_logp, ref_emb):
        logp_cur = dist.log_prob(ref_emb).sum(dim=-1)
        #logp_cur = F.log_softmax(dist.log_prob(ref_emb), dim=-1).sum(dim=-1)
        reward = logp_cur - ref_logp
        return reward
    def _compute_user_head_weight(self):
        origin_user_emb = self.user_embeddings.weight
        return origin_user_emb
    def user_head(self, input):
        linear_matrix = self._compute_user_head_weight()
        logist = torch.matmul(input, linear_matrix.T)
        return logist

    # def forward(self, input_ids_prompt, attention_mask, pos_emb, neg_emb, logp_ref_pos, logp_ref_neg, **kwargs):
    #     prompt_rep = self.encode_prompt(input_ids_prompt, attention_mask, **kwargs)
    #
    #     cur_mean = self.cur_mean_mapper(prompt_rep)
    #
    #     # 使用固定的 std
    #     #fixed_std_tensor = torch.full_like(cur_mean, self.fixed_std).to(attention_mask.device)
    #     dist = Normal(loc=cur_mean, scale=self.fixed_std)
    #
    #     # 计算正负 reward
    #     pos_reward = self.reward_funtion(dist, logp_ref_pos, pos_emb)
    #     neg_reward = self.reward_funtion(dist, logp_ref_neg, neg_emb)
    #
    #     loss = torch.mean(torch.nn.functional.softplus(pos_reward - neg_reward))
    #
    #     return {'loss': loss}

    def contrastive_loss(self, anchor_emb, pos_emb, neg_emb):
        """
        计算对比损失：拉近正样本，拉远负样本。
        """
        # 计算欧几里得距离
        pos_distance = F.pairwise_distance(anchor_emb, pos_emb, p=2)  # 正样本距离
        neg_distance = F.pairwise_distance(anchor_emb, neg_emb, p=2)  # 负样本距离

        # 对比损失（拉近正样本距离，拉远负样本距离）
        loss = 0.5 * (pos_distance.pow(2) + F.relu(self.margin - neg_distance).pow(2)).mean()
        return loss

    # def margin_ranking_loss(self,anchor, pos, neg, margin=1.0):
    #     pos_score = F.cosine_similarity(anchor, pos)
    #     neg_score = F.cosine_similarity(anchor, neg)
    #     target = torch.ones_like(pos_score)
    #     return F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
    # def triplet_loss(self, anchor, positive, negative, margin=0.5):
    #     pos_dist = F.pairwise_distance(anchor, positive, p=2)
    #     neg_dist = F.pairwise_distance(anchor, negative, p=2)
    #     loss = F.relu(pos_dist - neg_dist + margin).mean()
    #     return loss

    def forward(self, input_ids_prompt, attention_mask, pos_emb, neg_emb, logp_ref_pos, logp_ref_neg, **kwargs):
        prompt_rep = self.encode_prompt(input_ids_prompt, attention_mask, **kwargs)

        cur_mean = self.cur_mean_mapper(prompt_rep)

        loss = self.triplet_loss(cur_mean, pos_emb, neg_emb)

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
        std = torch.full_like(mean, self.fixed_std).to(attention_mask.device)
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
        #std = torch.full_like(mean, self.fixed_std).to(attention_mask.device)
        dist = Normal(loc=mean, scale=0.06)  # (B, 1, D)
        #dist2 = Normal(loc=mean, scale=0.6)

        # ---- 拼接 mean + samples ----
        all_embs = torch.cat([
            mean.unsqueeze(dim=0),  # (B, 1, D) -> mean embedding
            dist.rsample((num_samples,)), # (num_samples, B, D)
        ], dim=0)
        all_embs = all_embs

        # ---- Log probs ----
        log_probs = dist.log_prob(all_embs).sum(dim=-1) # (num_samples+1, B)

        # ---- Scores & Actions ----
        scores = torch.matmul(all_embs, self.user_embeddings.weight.T)  # (num_samples+1, B, num_users)
        _, topk_actions = torch.topk(scores, k=20, dim=-1)  # (num_samples+1, B, 20)

        return topk_actions, all_embs, log_probs

    def dpo_emb_sampling3(self, input_ids, attention_mask, num_samples, **kwargs):
        last_token_hidden_states = self.encode_prompt(input_ids, attention_mask, **kwargs)
        mean = self.cur_mean_mapper(last_token_hidden_states)  # (B, 1, D)
        std = torch.full_like(mean, self.fixed_std).to(attention_mask.device)
        dist = Normal(loc=mean, scale=std)  # (B, 1, D)

        sample = torch.randn_like(mean).to(mean.device) * 1 + mean

        # ---- 拼接 mean + samples ----
        all_embs = torch.cat([
            #mean.unsqueeze(dim=0),  # (B, 1, D) -> mean embedding
            #sample.unsqueeze(dim=0),  # (B, 1, D) -> mean embedding
            dist.rsample((num_samples,)).squeeze(2),  # (num_samples, B, D)
            #(torch.randn_like(mean).to(mean.device) * 0.6 + mean).unsqueeze(dim=0),
            (torch.randn_like(mean).to(mean.device) * 0.4 + mean).unsqueeze(dim=0),
            #(torch.randn_like(mean).to(mean.device) * 0.8 + mean).unsqueeze(dim=0),
        ], dim=0)
        all_embs = all_embs

        # ---- Log probs ----
        log_probs = dist.log_prob(all_embs).sum(dim=-1) # (num_samples+1, B)

        # ---- Scores & Actions ----
        scores = torch.matmul(all_embs, self.user_embeddings.weight.T)  # (num_samples+1, B, num_users)
        _, topk_actions = torch.topk(scores, k=20, dim=-1)  # (num_samples+1, B, 20)

        return topk_actions, all_embs, log_probs

    def init_model(self, path):
        # Load the state dict from the file
        self.cur_encoder = PeftModel.from_pretrained(self.cur_encoder, path)
        custom_weights = torch.load(os.path.join(path, "custom_weights.pt"), map_location=torch.device('cpu'))
        self.cur_mean_mapper.load_state_dict(custom_weights["mapper"])
        #self.cur_mean_mapper.load_state_dict(custom_weights["mean_mapper"])
        self.user_embeddings.load_state_dict(custom_weights["user_embeddings"])

    def init_model2(self, path):
        self.cur_encoder = PeftModel.from_pretrained(self.cur_encoder, path)
        custom_weights = torch.load(os.path.join(path, "custom_weights.pt"), map_location=torch.device('cpu'))
        self.cur_mean_mapper.load_state_dict(custom_weights["mean_mapper"])
        self.cur_std_mapper.load_state_dict(custom_weights["std_mapper"])
        self.user_embeddings.load_state_dict(custom_weights["user_embeddings"])

    def predict(self, item_id, input_ids_prompt, attention_mask, **kwargs):
        prompt_rep = self.encode_prompt(input_ids_prompt, attention_mask, **kwargs)
        cur_mean = self.cur_mean_mapper(prompt_rep)
        item_logist = self.user_head(cur_mean)
        _,topk_user = torch.topk(item_logist, 20)

        return item_id, topk_user

