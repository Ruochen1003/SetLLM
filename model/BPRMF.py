import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, reg_weight):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        #self.user_emb.weight.requires_grad= False
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.reg_weight = reg_weight

    def forward(self, user, pos_item, neg_item):
        user_e = self.user_emb(user)
        pos_e = self.item_emb(pos_item)
        neg_e = self.item_emb(neg_item)
        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)
        return pos_scores, neg_scores

    def predict(self, user, item):
        user_e = self.user_emb(user)
        item_e = self.item_emb(item)
        return (user_e * item_e).sum(dim=1)

    def getItemEmbedding(self, items):
        item_emb = self.item_emb(items)
        return item_emb

    def getUserEmbedding(self, users):
        user_emb = self.user_emb(users)
        return user_emb

    def item_bpr_loss(self, items, pos, neg):
        items_emb = self.item_emb(items)
        pos_emb = self.user_emb(pos)
        neg_emb = self.user_emb(neg)

        reg_loss = (1 / 2) * (items_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(items))

        pos_scores = torch.mul(items_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(items_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        loss = loss + reg_loss * self.reg_weight

        return loss

    def getUsersRating(self, users):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb.weight
        rating = torch.matmul(user_emb, item_emb.t())
        return rating

    def getItemsRating(self, items):
        item_emb = self.item_emb(items)
        user_emb = self.user_emb.weight
        rating = torch.matmul(item_emb, user_emb.t())
        return rating

    def inital_model(self, path):
        user_emb = torch.load(path + '/init_user_weight.pt')
        item_emb = torch.load(path + '/init_item_weight.pt')
        self.user_emb.weight.data.copy_(user_emb)
        self.item_emb.weight.data.copy_(item_emb)

    def inital_model2(self, path):
        user_emb = torch.load(path + '/merged_init_user_weight.pt')
        item_emb = torch.load(path + '/merged_init_item_weight.pt')
        self.user_emb.weight.data.copy_(user_emb)
        self.item_emb.weight.data.copy_(item_emb)
