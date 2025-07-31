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
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.num_users = user_num

        self.user_embeddings = nn.Embedding(user_num, config.hidden_size)
        self.margin = 1

    def reward_funtion(self, dist, ref_logp, ref_emb):
        logp_cur = dist.log_prob(ref_emb).sum(dim=-1)
        #logp_cur = F.log_softmax(dist.log_prob(ref_emb), dim=-1).sum(dim=-1)
        reward = logp_cur - ref_logp
        return reward
    def user_head(self, input):
        linear_matrix = self._compute_user_head_weight()
        logist = torch.matmul(input, linear_matrix.T)
        return logist

    def _compute_user_head_weight(self):
        origin_user_emb = self.user_embeddings.weight
        return origin_user_emb


    def forward(self, input_ids_prompt, attention_mask, pos_actions, neg_actions, **kwargs):
        prompt_rep = self.encode_prompt(input_ids_prompt, attention_mask, **kwargs)

        cur_mean = self.cur_mean_mapper(prompt_rep)
        item_logist = self.user_head(cur_mean)
        multi_user_label = self.get_multi_hot_label(pos_actions)
        n = multi_user_label.sum(dim=-1, keepdim=True)
        loss = F.cross_entropy(item_logist, multi_user_label.float() / n)

        # pos_emb_mean = self.user_embeddings(pos_actions).mean(dim=1)
        # pos_score = torch.matmul(cur_mean, pos_emb_mean.T).mean(dim=-1)
        #
        # neg_emb_mean = self.user_embeddings(neg_actions).mean(dim=1)
        # neg_score = torch.matmul(cur_mean, neg_emb_mean.T).mean(dim=-1)
        #
        # loss = -F.logsigmoid(pos_score - neg_score).mean()

        return {'loss': loss}

    def encode_prompt(self, input_ids_prompt, attention_mask,  **kwargs):
        outputs_main = self.cur_encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True, **kwargs)
        last_hidden_states = outputs_main.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[:, -1, :]
        return last_token_hidden_states

    def dpo_emb_sampling2(self, input_ids, attention_mask, num_samples, **kwargs):
        last_token_hidden_states = self.encode_prompt(input_ids, attention_mask, **kwargs)
        mean = self.cur_mean_mapper(last_token_hidden_states)  # (B, 1, D)
        dist = Normal(loc=mean, scale=0.06)  # (B, 1, D)

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

    def get_multi_hot_label(self, labels):
        labels = labels + 1
        multi_label = torch.zeros((len(labels), self.num_users + 1), dtype=torch.bool).to(labels.device)
        multi_label.scatter_(1, labels, True)
        multi_label = multi_label[:, 1:]
        return multi_label

