import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset

# import cpp_sampler

class BPRDataset_update(Dataset):
    def __init__(self, item, sampled_users, emb, prob, ground_items, all_interacted_user_ids):
        self.emb = emb
        self.prob = prob
        self.sampled_action = sampled_users
        #self.all_interaction = sampled_users
        self.all_interaction = all_interacted_user_ids + sampled_users.tolist()
        self.item = torch.cat([ground_items, item])
        #self.item = item

    def __getitem__(self, index):
        item = self.item[index]
        sampled_users = self.all_interaction[index]

        return item, sampled_users
    def __len__(self):
        return len(self.item)

    def user_items(self):
        """
        返回 dict: {user_id: [item1, item2, ...]}
        """
        user2items = defaultdict(list)
        item_np = self.item.cpu().numpy()

        for item_id, users in zip(item_np, self.all_interaction):
            for user in users:
                user2items[user].append(item_id)

        return dict(user2items)


class BPRDataCollator:
    def __init__(self, num_users, num_negatives=1):
        """
        Args:
            num_users (int): 总用户数量
            num_negatives (int): 每个正样本对应的负样本数量，默认1
        """
        self.num_users = num_users
        self.num_negatives = num_negatives
    def __call__(self, batch):
        item_list = []
        pos_user_list = []
        neg_user_list = []

        for item, sampled_users in batch:
            if isinstance(sampled_users, torch.Tensor):
                sampled_users_list = sampled_users.tolist()
            else:
                sampled_users_list = sampled_users

            pos_user = random.choice(sampled_users_list)
            neg_users = []

            while len(neg_users) < self.num_negatives:
                neg_user = random.randint(0, self.num_users - 1)
                if neg_user not in sampled_users_list:
                    neg_users.append(neg_user)

            item_list.append(item)
            pos_user_list.append(pos_user)
            neg_user_list.append(neg_users if self.num_negatives > 1 else neg_users[0])
        return (
            torch.stack(item_list),
            torch.tensor(pos_user_list, dtype=torch.long),
            torch.tensor(neg_user_list, dtype=torch.long),
        )


        # processed_batch = []
        #
        # for item, users in batch:
        #     if isinstance(users, torch.Tensor):
        #         users = users.tolist()
        #     processed_batch.append((item, users))
        #
        # item_list, pos_user_list, neg_user_list = cpp_sampler.sample_batch(
        #     processed_batch,
        #     num_users=self.num_users,
        #     num_negatives=self.num_negatives
        # )
        # # 3. 转为 tensor，注意 neg_user_list 是 list[list[int]]
        # item_tensor = torch.tensor(item_list, dtype=torch.long)
        # pos_user_tensor = torch.tensor(pos_user_list, dtype=torch.long)
        # if self.num_negatives == 1:
        #     # 每个样本只采样一个负例，neg_user_list 是 list[int]
        #     neg_user_tensor = torch.tensor([neg[0] for neg in neg_user_list], dtype=torch.long)
        # else:
        #     # 每个样本多个负例，neg_user_list 是 list[list[int]]
        #     neg_user_tensor = torch.tensor(neg_user_list, dtype=torch.long)
        # return item_tensor, pos_user_tensor, neg_user_tensor


