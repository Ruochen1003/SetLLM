import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset

try:
    import cpp_sampler
    sample_ext = True
except:
    sample_ext = False

class BPRDataset(Dataset):
    def __init__(self, item, sampled_users, emb, prob, ):
        self.item = item
        self.sampled_users = sampled_users
        self.emb = emb
        self.prob = prob
    def __getitem__(self, index):
        item = self.item[index]
        sampled_users = self.sampled_users[index]

        return item, sampled_users
    def __len__(self):
        return len(self.item)

    def user_items(self):
        """
        返回 dict: {user_id: [item1, item2, ...]}
        """
        user2items = defaultdict(list)
        item_np = self.item.cpu().numpy()
        users_np = self.sampled_users.cpu().numpy()

        for item_id, users in zip(item_np, users_np):
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

        # for item, sampled_users in batch:
        #     if isinstance(sampled_users, torch.Tensor):
        #         sampled_users_list = sampled_users.tolist()
        #     else:
        #         sampled_users_list = sampled_users
        #
        #     pos_user = random.choice(sampled_users_list)
        #     neg_users = []
        #
        #     while len(neg_users) < self.num_negatives:
        #         neg_user = random.randint(0, self.num_users - 1)
        #         if neg_user not in sampled_users_list:
        #             neg_users.append(neg_user)
        #
        #     item_list.append(item)
        #     pos_user_list.append(pos_user)
        #     neg_user_list.append(neg_users if self.num_negatives > 1 else neg_users[0])
        processed_batch = []

        for item, users in batch:
            if isinstance(users, torch.Tensor):
                users = users.tolist()
            processed_batch.append((item, users))

        item_list, pos_user_list, neg_user_list = cpp_sampler.sample_batch(
            processed_batch,
            num_users=self.num_users,
            num_negatives=self.num_negatives
        )

        return (
            torch.stack(item_list),
            torch.tensor(pos_user_list, dtype=torch.long),
            torch.tensor(neg_user_list, dtype=torch.long),
        )
