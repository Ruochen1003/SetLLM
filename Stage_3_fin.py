import copy
import pickle
import argparse

import torch
import os

from torch import nn
from torch.utils.data import DataLoader

from Dataset.RecDataset import RecDataset, RecDataCollator
from Dataset.BPRDateset_update import BPRDataCollator
from Dataset.DPODataset import DPOCollator
from model.BPRMF import BPRMF
from model.FilterLLMS3_fixed import FilterLLM_S3

from utils.utils import test_per_item, dpo_sample, generate_bpr_training_dataset_update, build_dpo_embedding_dataset, \
    test_per_item2, test_per_item3, generate_bpr_training_dataset2, build_dpo_embedding_dataset_update
from utils.training import bprmf_training, dpo_training
from utils.setup import setup_environment, get_llama_model, get_tokenizer, get_dpo_training_cofig
import gc

def release_memory():
    torch.cuda.empty_cache()
    gc.collect()


def removed_mask(tensor: torch.Tensor) -> torch.Tensor:

    # 每行的最小值与最大值
    row_min, _ = tensor.min(dim=1)
    row_max, _ = tensor.max(dim=1)

    # 如果最小值 == 最大值，说明该行所有值都相同
    uniform_mask = (row_min == row_max)

    return ~uniform_mask


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    accelerator = setup_environment(args)

    # === Dataset Info ===
    dataset_root = os.path.join('./data', args.dataset)
    convert_dict = pickle.load(open(os.path.join(dataset_root, 'convert_dict.pkl'), 'rb'))
    num_users = max(convert_dict['user_array']) + 1
    num_items = max(convert_dict['item_array']) + 1
    accelerator.print(f"Dataset loaded: {args.dataset} with {num_users} users, {num_items} items")

    # === Tokenizer & Collators ===
    tokenizer = get_tokenizer(args)
    data_collator = RecDataCollator(tokenizer)
    bprcollator = BPRDataCollator(num_users)
    dpocollator = DPOCollator(tokenizer)

    # === Load Cold Dataset ===
    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    cold_file_path = os.path.join(dataset_root, 'warm_emb.csv')
    cold_dataset = RecDataset(content_file_path, [cold_file_path], args.dataset, max_length=args.max_length)
    all_item_ids, all_interacted_user_ids = cold_dataset.get_item_groundTrue()
    cold_dataloader = DataLoader(cold_dataset, batch_size=args.sample_batch, collate_fn=data_collator)
    cold_dataloader = accelerator.prepare(cold_dataloader)

    # === Initialize Sample Model ===
    inital_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_2")
    llama_model, config = get_llama_model(args)
    sample_model = FilterLLM_S3(config, llama_model, num_users)
    sample_model.init_model(inital_path)
    sample_model = accelerator.prepare(sample_model)

    for dpo_time in range(args.dpo_times):
        # === Sampling from LLM Model ===
        with torch.no_grad():
            accelerator.print("Sampling from model...")
            item_ids, sampled_action, sampled_emb, sampled_action_probs = dpo_sample(
                sample_model, cold_dataloader, args.sample_num)
            item_ids = torch.cat(item_ids, dim=0).cpu()
            sampled_action = torch.cat(sampled_action, dim=1).cpu()
            sampled_emb = torch.cat(sampled_emb, dim=1).cpu()
            sampled_action_probs = torch.cat(sampled_action_probs, dim=1).cpu()

        # === Prepare for Evaluation ===
        base_model = BPRMF(num_users, num_items, emb_size=200, reg_weight=0.01)
        train_item_index, all_val_result, all_sampled_emb, all_sampled_action_probs = [], [], [], []

        for i in range(2):
            test_item_num = int(len(all_item_ids) * args.test_ratio)
            perm = torch.randperm(len(all_item_ids))
            test_index = perm[:test_item_num]
            ground_indices = perm[test_item_num:]
            test_interacted_user = [all_interacted_user_ids[i] for i in test_index]

            accelerator.print(f"-------------- DPO Sample {dpo_time} --------------")
            training_data_list = generate_bpr_training_dataset_update(
                item_ids, sampled_action, sampled_emb, sampled_action_probs,
                test_index, all_interacted_user_ids, ground_indices
            )

            ndcg_result, score_result = [], []
            accelerator.print("Training BPR model...")

            for index, data in enumerate(training_data_list):
                accelerator.print(f"Training BPR model with dataset {index}...")
                model = copy.deepcopy(base_model)
                model, data = accelerator.prepare(model, data)
                model = bprmf_training(model, data, accelerator, bprcollator, args.bpr_batch_size, epochs=args.bpr_num_epochs)

                with torch.no_grad():
                    ndcg, score = test_per_item3(model, item_ids[test_index], test_interacted_user, args, accelerator.device)
                    ndcg_result.append(ndcg)
                    score_result.append(score)

                del model, data
                release_memory()

            # === Evaluation Metrics ===
            # ndcg_result = torch.stack(ndcg_result, dim=0)
            score_result = torch.stack(score_result, dim=0)
            # mask = (ndcg_result.sum(dim=0) == 0).unsqueeze(0)
            # val_result = torch.where(mask, score_result, ndcg_result)
            val_result = score_result

            sampled_emb_list = [data.emb for data in training_data_list]
            sampled_action_probs_list = [data.prob for data in training_data_list]
            train_item_index.append(item_ids[test_index])
            all_val_result.append(val_result)
            all_sampled_emb.append(torch.stack(sampled_emb_list))
            all_sampled_action_probs.append(torch.stack(sampled_action_probs_list))

        # === Build DPO Dataset ===
        if args.save_data:
            save_train_path = os.path.join('data', args.dataset, f'{args.LLM_type}_{dpo_time}_dpo_train_data.pt')
            save_test_path = os.path.join('data', args.dataset, f'{args.LLM_type}_{dpo_time}_dpo_test_data.pt')
        else:
            save_train_path = save_test_path = None

        dpo_train_dataset, dpo_test_dataset = build_dpo_embedding_dataset_update(
            train_item_index, all_val_result, all_sampled_emb, all_sampled_action_probs,
            content_file_path, args.dataset, top_k=args.topks,
            save_train_path=save_train_path, save_test_path=save_test_path
        )

        del train_item_index, all_val_result, all_sampled_emb, all_sampled_action_probs
        del sampled_emb, sampled_action_probs, training_data_list
        release_memory()

        # === DPO Fine-tuning ===
        training_args = get_dpo_training_cofig(args)
        sample_model = dpo_training(sample_model, dpo_train_dataset, dpo_test_dataset, dpocollator, training_args)

        del dpo_train_dataset, dpo_test_dataset
        release_memory()

        # === Save Model ===
        save_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_3")
        os.makedirs(save_path, exist_ok=True)
        sample_model.cur_encoder.save_pretrained(save_path)
        torch.save(
            {
                "mean_mapper": sample_model.cur_mean_mapper.state_dict(),
                "user_embeddings": sample_model.user_embeddings.state_dict(),
            },
            os.path.join(save_path, "custom_weights.pt"),
        )


# other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
#parser.add_argument('--gpu_id', type=str, default='4,5')

# dataset setting
parser.add_argument("--dataset", type=str, default='CiteULike', help="specify the dataset for experiment")

# LLM setting
parser.add_argument('--LLM_root', type=str, default="./LLM_base")
parser.add_argument('--LLM_type', type=str, default="Llama3-1B", help='LLM model type (Llama2-7B, GPT2, Llama3-1B, Llama3-3B,  Llama3-13B)')
parser.add_argument("--max_length", type= int, default=512, help='max taken length of LLM')

# sample setting
parser.add_argument('--sample_batch', type=int, default=16, help="batch size")
parser.add_argument('--sample_num', type=int, default=7, help="sample number of each item")

# training details
parser.add_argument('--dpo_batch_size', type=int, default=16, help="batch size")
parser.add_argument('--bpr_batch_size', type=int, default=4096, help="batch size")

# test details
parser.add_argument('--predict_batch_size', type=int, default=16, help="batch size")
parser.add_argument('--test_batch', type=int, default=100, help="batch size")
parser.add_argument('--bpr_num_epochs', type=int, default=500)

parser.add_argument('--dpo_num_epochs', type=int, default=1)
parser.add_argument('--dpo_times', type=int, default=1)
parser.add_argument('--topks', type=int, default=2, help="topk")

parser.add_argument('--save_data', type=bool, default=False)
parser.add_argument("--dpo_lr", type=float, default=1e-5, help="learning rate")

parser.add_argument("--test_ratio", type=float, default=0.2, help="test ratio")


args = parser.parse_args()

if __name__ == "__main__":
    main(args)

