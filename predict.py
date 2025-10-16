import pickle
import argparse
import random

import numpy as np
import torch
import os
from accelerate import Accelerator
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from Dataset.RecDataset import RecDataset, RecDataCollator

from model.FilterLLMS2 import FilterLLM

import pandas as pd
from peft import PeftModel



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(cpu=False)
    if torch.cuda.is_available():
        print(f"device: {accelerator.device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {args.dataset}")

    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    accelerator.print(f'process: {accelerator.num_processes}')

    #TODO complete the output training information,user number, item number
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    dataset_root = os.path.join('./data', args.dataset)
    num_user_item_path = os.path.join(dataset_root, 'convert_dict.pkl')
    convert_dict = pickle.load(open(num_user_item_path,'rb'))
    num_users= max(convert_dict['user_array']) + 1
    num_items= max(convert_dict['item_array']) + 1
    origin_dim = 200

    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    if args.LLM_type =='LLama2':
        llm_root = os.path.join(args.LLM_root, args.LLM_type)
    elif args.LLM_type == 'Llama3-1B':
        llm_root = '/home/models/Llama3-1B'
    elif args.LLM_type == "Llama3-3B":
        llm_root = '/home/models/Llama-3.2-3B'
    elif args.LLM_type == "Llama3-8B":
        llm_root = '/root/autodl-tmp/model'
    elif args.LLM_type == "Llama3-13B":
        llm_root = '/home/models/Llama-2-13b-hf'
    else:
        raise Exception("LLM_type don't exist")
    tokenizer = AutoTokenizer.from_pretrained(llm_root,padding_side = "left")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = RecDataCollator(tokenizer)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")

    '''
        Instantiate the pretrained Llama model
    '''
    # TODO need extension for llama
    accelerator.print(f"-----Begin Instantiating the Pretrained {args.LLM_type} Model-----")

    LlamaModel = LlamaForCausalLM.from_pretrained(llm_root, attn_implementation="eager")
    config = LlamaModel.config
    config.hidden_layers = args.hidden_layers
    # config.num_users = num_users
    # config.num_items = num_items
    config.origin_dim = origin_dim
    config.lamda = args.lamda
    accelerator.print("Success!")
    accelerator.print(f"-----End Instantiating the Pretrained {args.LLM_type} Model-----\n")
    '''
        Instantiate the llama for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Pairwise Recommendation Model-----")



    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pairwise Recommendation Model-----\n")
    inference_args = TrainingArguments(
        output_dir=f'data/{args.dataset}',
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=args.predict_batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16_full_eval=True,
    )
    rec_model = FilterLLM(config, LlamaModel, num_users, num_items)
    if args.stage == 1:
        weight_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_1")
    elif args.stage == 2:
        weight_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_2")
        custom_weights = torch.load(os.path.join(weight_path, "custom_weights.pt"))
        rec_model.mapper.load_state_dict(custom_weights["mapper"])
        rec_model.user_embeddings.load_state_dict(custom_weights["user_embeddings"])
    elif args.stage == 3:
        weight_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_3")
        custom_weights = torch.load(os.path.join(weight_path, "custom_weights.pt"))
        rec_model.mapper.load_state_dict(custom_weights["mean_mapper"])
        rec_model.user_embeddings.load_state_dict(custom_weights["user_embeddings"])
    else:
        Exception("stage don't exist")

    rec_model.encoder = PeftModel.from_pretrained(LlamaModel, weight_path)

    dataset_root = os.path.join('./data', args.dataset)
    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    cold_val_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    cold_test_file_path2 = os.path.join(dataset_root, 'cold_item_test.csv')
    cold_item_dataset = RecDataset(content_file_path, [cold_val_file_path,cold_test_file_path2],
                                   args.dataset, max_length=args.max_length)



    trainer = Trainer(model=rec_model, args=inference_args, data_collator = data_collator)

    output = trainer.predict(cold_item_dataset)
    small_emb_path = os.path.join(dataset_root, 'bprmf_ALDI.npy')
    small_emb = np.load(small_emb_path)
    small_emb = torch.tensor(small_emb).to(accelerator.device)
    user_emb = small_emb[:num_users]
    item_emb = small_emb[num_users:]
    update_dataset(output)


    #update_dataset2(output, item_emb, user_emb, 1, args)


def update_dataset(output):
    item_list = output.predictions[2]
    predict_user_list = output.predictions[1]

    user_item_pairs = []

    for batch_item_list, batch_predict_top_k in zip(item_list, predict_user_list):
        for user_id in batch_predict_top_k:
            user_item_pairs.append([user_id.item(), batch_item_list.item()])

    df = pd.DataFrame(user_item_pairs, columns=["user", "item"])

    dataset_root = os.path.join('./data', args.dataset)
    saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}_predicted_cold_item_interaction.csv')

    df.to_csv(saved_file_path, index=False)
    print(f"User-item pairs saved to {saved_file_path}")

def update_dataset2(output, item_emb, user_emb, ratio, args):
    item_list = output.predictions[2]  # list of item ids
    logist = torch.tensor(output.predictions[0]).to(user_emb.device)   # shape: [batch_size, num_users]

    user_item_pairs = []

    item_emb = torch.tensor(item_emb).to(user_emb.device) if not isinstance(item_emb, torch.Tensor) else item_emb.to(user_emb.device)
    user_emb = torch.tensor(user_emb).to(user_emb.device) if not isinstance(user_emb, torch.Tensor) else user_emb.to(user_emb.device)

    # Compute similarity: [num_items, num_users]
    sim_scores = torch.matmul(item_emb, user_emb.T)  # [num_items, num_users]

    for idx, (item_id, item_logits) in enumerate(zip(item_list, logist)):
        item_sim = sim_scores[item_id]  # [num_users]

        # Top N most similar users
        num_top_sim_users = max(1, int(ratio * len(user_emb)))
        top_sim_user_indices = torch.topk(item_sim, num_top_sim_users).indices  # user ids

        # Get logits for those users
        logits_top_sim_users = item_logits[top_sim_user_indices]  # [num_top_sim_users]

        # Top K logits within this subset
        num_top_logits = min(20, len(logits_top_sim_users))
        top_k_within_subset = torch.topk(logits_top_sim_users, num_top_logits)

        # 获取最终的 global user id（从 top_sim_user_indices 中选出 top-k）
        final_user_ids = top_sim_user_indices[top_k_within_subset.indices]

        # Add (user, item) pairs
        for user_id in final_user_ids.tolist():
            user_item_pairs.append((user_id, item_id))

    df = pd.DataFrame(user_item_pairs, columns=["user", "item"])

    dataset_root = os.path.join('./data', args.dataset)
    os.makedirs(dataset_root, exist_ok=True)
    saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}_predicted_cold_item_interaction_{ratio}.csv')

    df.to_csv(saved_file_path, index=False)
    print(f"User-item pairs saved to {saved_file_path}")

def update_dataset2_random(output, user_emb, ratio, args, top_k=20, seed=42):
    item_list = output.predictions[2]  # list of item ids
    logist = torch.tensor(output.predictions[0]).to(user_emb.device)   # shape: [batch_size, num_users]

    user_item_pairs = []
    #user_emb = torch.tensor(user_emb).to(user_emb.device) if not isinstance(user_emb, torch.Tensor) else user_emb.to(user_emb.device)

    num_users = user_emb.shape[0]
    num_random_users = max(1, int(ratio * num_users))  # 抽样用户数

    random.seed(seed)

    for idx, (item_id, item_logits) in enumerate(zip(item_list, logist)):
        # 1. 随机选取一部分用户
        all_user_indices = list(range(num_users))
        sampled_user_indices = random.sample(all_user_indices, num_random_users)  # list[int]
        sampled_user_indices = torch.tensor(sampled_user_indices, device=user_emb.device)

        # 2. 从 sampled 中选 logits top-k
        sampled_logits = item_logits[sampled_user_indices]
        num_top_logits = min(top_k, len(sampled_logits))
        topk_in_sample_indices = torch.topk(sampled_logits, num_top_logits).indices
        final_user_ids = sampled_user_indices[topk_in_sample_indices]

        # 3. 保存 (user_id, item_id)
        for user_id in final_user_ids.tolist():
            user_item_pairs.append((user_id, item_id))

    # 保存为 CSV
    df = pd.DataFrame(user_item_pairs, columns=["user", "item"])
    dataset_root = os.path.join('./data', args.dataset)
    os.makedirs(dataset_root, exist_ok=True)
    saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}_predicted_cold_item_interaction_random_{ratio}.csv')

    df.to_csv(saved_file_path, index=False)
    print(f"User-item pairs saved to {saved_file_path}")



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

# recommendation setting
parser.add_argument('--list_length', type=int, default=40, help='number of retrivaled users')

# training details
parser.add_argument('--pretrain_batch_size', type=int, default=8)
parser.add_argument('--finetune_batch_size', type=int, default=1)
parser.add_argument('--predict_batch_size',  type=int, default=8)

# backbone
parser.add_argument('--backbone_type', type=str, default='LightGCN')
parser.add_argument('--hidden_layers', type=list, default=[512, 256, 512], help='hidden_layers of mapper')

parser.add_argument('--num_pretrained_epochs', type=int, default=100) #如果使用多分类，10轮就足够了
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lamda', type=int, default=10)

parser.add_argument('--pretrain', type=int, default=1, help='0 for predict only, 1 for pretraining only, 2 for fine-tuning only, 3 for both')

parser.add_argument('--predict_way', type=str, default='positive', help='positive, negative, both')
parser.add_argument('--full_model_update', type=str, default='lora finetune', help='lora finetune or full finetune')
parser.add_argument('--stage', type=int, default=2, help='1 for stage 1, 2 for stage 2, 3 for stage 3')


#TODO


args = parser.parse_args()

if __name__ == "__main__":
    main(args)
