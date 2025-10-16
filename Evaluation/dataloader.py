"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
import pickle
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path=None):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/warm_emb.csv'
        self.LLM = config['LLM']
        extended_file = path + '/' + config['LLM'] + '_predicted_cold_item_interaction.csv'
        test_file = path + '/warm_test.csv'
        n_user_item_file = path + '/n_user_item.pkl'
        self.para_dict = pickle.load(open(path + '/convert_dict.pkl', 'rb'))

        n_user_item = pickle.load(open(n_user_item_file, 'rb'))
        self.test_batch = config['test_u_batch_size']
        self.n_user = n_user_item['user']
        self.m_item = n_user_item['item']
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for n, l in enumerate(f.readlines()):
                if len(l) > 0 and n > 0:
                    l = l.strip('\n').split(',')
                    trainUser.append(int(l[0]))
                    trainItem.append(int(l[1]))
                    trainUniqueUsers.append(int(l[0]))
                    self.traindataSize += 1

        # Add predicted interaction into training data
        with open(extended_file) as f:
            for n, l in enumerate(f.readlines()):
                if len(l) > 0 and n > 0:
                    l = l.strip('\n').split(',')
                    trainUser.append(int(l[0]))
                    trainItem.append(int(l[1]))
                    trainUniqueUsers.append(int(l[0]))
                    self.traindataSize += 1

        self.trainUniqueUsers = np.array(set(trainUniqueUsers))
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for n, l in enumerate(f.readlines()):
                if len(l) > 0 and n > 0:
                    l = l.strip('\n').split(',')
                    testUser.append(int(l[0]))
                    testItem.append(int(l[1]))
                    testUniqueUsers.append(int(l[0]))
                    self.testDataSize += 1
        self.testUniqueUsers = np.array(set(testUniqueUsers))
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        # print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.all_exclude_pair()

        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + f'/{self.LLM}_fin_s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + f'/{self.LLM}_fin_s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getItemPosUsers(self, items):
        posUsers = []
        for item in items:
            posUsers.append(self.UserItemNet[:, item].nonzero()[0])
        return posUsers

    def get_exclude_pair_count(self, ts_user, ts_nei, batch):
        exclude_pair_list = []
        exclude_count = [0]  # 每个 user 有多少 exclude pair
        for i, beg in enumerate(range(0, len(ts_user), batch)):
            end = min(beg + batch, len(ts_user))
            batch_user = ts_user[beg:end]
            batch_range = list(range(end - beg))
            batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)

            specialize_get_exclude_pair = lambda x: self.get_exclude_pair(x, ts_nei)
            exclude_pair = list(map(specialize_get_exclude_pair, batch_u_pair))
            exclude_pair = np.concatenate(exclude_pair, axis=0)

            exclude_pair_list.append(exclude_pair)
            exclude_count.append(exclude_count[i] + len(exclude_pair))

        exclude_pair_list = np.concatenate(exclude_pair_list, axis=0)
        return [exclude_pair_list, exclude_count]

    def get_exclude_pair(self, u_pair, ts_nei):
        pos_item = np.array(sorted(list(set(self.para_dict['pos_user_nb'][u_pair[0]]) - set(ts_nei[u_pair[0]]))),
                            dtype=np.int64)
        pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
        return np.stack([pos_user, pos_item], axis=1)

    def all_exclude_pair(self):
        self.exclude_val_warm = self.get_exclude_pair_count(self.para_dict['warm_val_user'],
                                                            self.para_dict['warm_val_user_nb'],
                                                            self.test_batch)
        self.exclude_val_cold = self.get_exclude_pair_count(self.para_dict['cold_val_user'],
                                                            self.para_dict['cold_val_user_nb'],
                                                            self.test_batch)
        self.exclude_val_overall = self.get_exclude_pair_count(self.para_dict['overall_val_user'],
                                                               self.para_dict['overall_val_user_nb'],
                                                               self.test_batch)
        self.exclude_test_warm = self.get_exclude_pair_count(self.para_dict['warm_test_user'],
                                                             self.para_dict['warm_test_user_nb'],
                                                             self.test_batch)
        self.exclude_test_cold = self.get_exclude_pair_count(self.para_dict['cold_test_user'],
                                                             self.para_dict['cold_test_user_nb'],
                                                             self.test_batch)
        self.exclude_test_overall = self.get_exclude_pair_count(self.para_dict['overall_test_user'],
                                                                self.para_dict['overall_test_user_nb'],
                                                                self.test_batch)

    def test_user_nb(self):
        return self.para_dict['cold_test_user_nb'], self.para_dict['warm_test_user_nb'], self.para_dict[
            'overall_test_user_nb']

    def test_user(self):
        return self.para_dict['cold_test_user'], self.para_dict['warm_test_user'], self.para_dict['overall_test_user']

    def test_exclude(self):
        return self.exclude_test_cold, self.exclude_test_warm, self.exclude_test_overall

    def val_user_nb(self):
        return self.para_dict['cold_val_user_nb'], self.para_dict['warm_val_user_nb'], self.para_dict[
            'overall_val_user_nb']

    def val_user(self):
        return self.para_dict['cold_val_user'], self.para_dict['warm_val_user'], self.para_dict['overall_val_user']

    def val_exclude(self):
        return self.exclude_val_cold, self.exclude_val_warm, self.exclude_val_overall

    def warm_cold_item(self):
        return self.para_dict['warm_item'], self.para_dict['cold_item']
