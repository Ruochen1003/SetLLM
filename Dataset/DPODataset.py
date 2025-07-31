import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding
class DPODataset(Dataset):
    def __init__(self, item_ids, score_list, embedding_list, prob_list, item_content_path, dataset_name, top_k=1):
        """
        Args:
            item_ids: List[Any], 长度为 n
            score_list: List[Tensor], 长度为 k，每个元素大小为 (n,)
            embedding_list: List[List[Tensor]], 长度为 k，每个元素是 n 个 embedding
            prob_list: List[Tensor], 长度为 k，每个元素大小为 (n,)
            top_k: int, 每个 item 取前 top_k 个采样方式作为 chosen
        """
        self.item_ids = item_ids
        self.top_k = top_k
        self.dataset_name = dataset_name
        if self.dataset_name == 'CiteULike':
            item_content = pd.read_csv(item_content_path, encoding='latin1')
        else:
            item_content = pd.read_csv(item_content_path)

        #self.scores_tensor = torch.stack(score_list)  # (k, n)
        self.scores_tensor = score_list  # (k, n)
        self.probs_tensor = torch.stack(prob_list)    # (k, n)
        self.embedding_list = torch.stack(embedding_list)
        self.remove_uniform_rows()
        self.n = len(self.item_ids)
        self.k = len(self.scores_tensor)
        self.prompt_list = self._item_content_process(self.item_ids, item_content)

        # 构造所有 (chosen, rejected) 对的索引列表
        self.pairs = []  # 每个元素 (item_idx, chosen_sample_idx, rejected_sample_idx)


        for i in range(self.n):
            item_scores = self.scores_tensor[:, i]  # (k,)
            sorted_indices = torch.argsort(item_scores, descending=True)
            chosen_indices = sorted_indices[:top_k]
            rejected_indices = sorted_indices[-top_k:]

            for chosen_idx in chosen_indices:
                for rejected_idx in rejected_indices:
                    self.pairs.append((i, chosen_idx.item(), rejected_idx.item()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_idx, chosen_idx, rejected_idx = self.pairs[idx]

        return {
            "item_id": self.item_ids[item_idx],
            "input_ids_prompt": self.prompt_list[item_idx],
            "pos_emb": self.embedding_list[chosen_idx][item_idx],
            "neg_emb": self.embedding_list[rejected_idx][item_idx],
            "logp_ref_pos": self.probs_tensor[chosen_idx, item_idx],
            "logp_ref_neg": self.probs_tensor[rejected_idx, item_idx]
        }

    def _item_content_process(self, item_ids, all_item_content):
        prompt_list = []
        if self.dataset_name == 'CiteULike':
            for item_id in item_ids:
                prompt_list.append(
                    f"Assume you are an expert in the field of recommendation. "
                    f"Please predict which readers a paper titled \"{all_item_content.loc[item_id].title}\" would attract."
                )
        elif self.dataset_name == 'Amazon_Beauty':
            for item_id in item_ids:
                title = f"title: {all_item_content.loc[item_id].title}" if pd.notna(all_item_content.loc[item_id].title) else ""
                brand = f"brand: {all_item_content.loc[item_id].brand}" if pd.notna(all_item_content.loc[item_id].brand) else ""
                description = f"description: {all_item_content.loc[item_id].description}" if pd.notna(all_item_content.loc[item_id].description) else ""
                prompt_list.append(
                    f"A beauty product has the following features: {title}, {brand}, {description}. "
                    f"Predict the potential customer for this product."
                )
        elif self.dataset_name == 'ml-10m':
            for item_id in item_ids:
                title = f"Title: {all_item_content.loc[item_id].Title}" if pd.notna(all_item_content.loc[item_id].Title) else ""
                genres = f"Genres: {all_item_content.loc[item_id].Genres}" if pd.notna(all_item_content.loc[item_id].Genres) else ""
                prompt_list.append(
                    f"Assume you are an expert in the field of recommendation. "
                    f"A movie has the following features: {title}, {genres}. Predict the potential customer for this movie."
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return prompt_list

    def removed_mask(self,tensor: torch.Tensor) -> torch.Tensor:

        # 每行的最小值与最大值
        col_min, _ = tensor.min(dim=0)
        col_max, _ = tensor.max(dim=0)

        # 如果最小值 == 最大值，说明该行所有值都相同
        uniform_mask = (col_min == col_max)

        return ~uniform_mask

    def remove_uniform_rows(self):
        removed_mask = self.removed_mask(self.scores_tensor)
        self.scores_tensor = self.scores_tensor[:, removed_mask]
        self.probs_tensor = self.probs_tensor[:, removed_mask]
        self.embedding_list = self.embedding_list[:, removed_mask, :]
        self.item_ids = self.item_ids[removed_mask].tolist()


class DPOCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_length=512, padding_value=-1):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, features):
        prompts = [f["input_ids_prompt"] for f in features]

        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Build the batch
        batch = {
            "input_ids_prompt": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pos_emb": torch.stack([f['pos_emb'].cpu() for f in features], dim=0),
            "neg_emb": torch.stack([f['neg_emb'].cpu() for f in features], dim=0),
            "logp_ref_pos": torch.stack([f["logp_ref_pos"].cpu() for f in features], dim=0),
            "logp_ref_neg": torch.stack([f["logp_ref_neg"].cpu() for f in features], dim=0)

        }

        # Include item_id if needed
        if "item_id" in features[0]:
            batch["item_id"] = torch.tensor([f["item_id"] for f in features])

        return batch


