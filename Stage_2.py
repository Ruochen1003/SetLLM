import pickle
import argparse

import torch
import os
from accelerate import Accelerator
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from Dataset.RecDataset import RecDataset, RecDataCollator
from model.FilterLLMS2 import FilterLLM
from peft import LoraConfig, TaskType, get_peft_model

from utils.setup import setup_environment, get_llama_model, get_tokenizer, get_dpo_training_cofig


def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    accelerator = setup_environment(args)

    accelerator.print("-----Current Setting-----")

    accelerator.print("-----Begin Obtaining Dataset Info-----")
    dataset_root = os.path.join('./data', args.dataset)
    num_user_item_path = os.path.join(dataset_root, 'convert_dict.pkl')
    convert_dict = pickle.load(open(num_user_item_path,'rb'))
    num_users= max(convert_dict['user_array']) + 1
    num_items= max(convert_dict['item_array']) + 1

    tokenizer = get_tokenizer(args)
    data_collator = RecDataCollator(tokenizer)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")

    '''
        Instantiate the pretrained Llama model
    '''
    # TODO need extension for llama
    accelerator.print(f"-----Begin Instantiating the Pretrained {args.LLM_type} Model-----")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    LlamaModel, config = get_llama_model(args)

    accelerator.print("Success!")
    accelerator.print(f"-----End Instantiating the Pretrained {args.LLM_type} Model-----\n")
    '''
        Instantiate the llama for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Pairwise Recommendation Model-----")
    rec_model = FilterLLM(config, LlamaModel, num_users, num_items)
    rec_model.encoder = get_peft_model(LlamaModel, lora_config)

    root = os.path.join('./model_weight', args.dataset, args.LLM_type, "stage_1")
    user_emb_path = os.path.join(root, 'init_user_weight.pt')
    accelerator.print(f"Load user embedding from {user_emb_path}")
    rec_model.load_user_emb(user_emb_path)

    item_emb_path = os.path.join(root, f'init_item_weight.pt')
    accelerator.print(f"Load item embedding from {item_emb_path}")
    rec_model.load_item_emb(item_emb_path)
    #map_path = os.path.join(root, 'init_map_weight.pt')
    #accelerator.print(f"Load item embedding from {map_path}")
    #rec_model.mapper.load_state_dict(torch.load(map_path))
    # weight_path = os.path.join('model_weight', args.dataset, args.LLM_type, "stage_2")
    # custom_weights = torch.load(os.path.join(weight_path, "custom_weights.pt"))
    # rec_model.mapper.load_state_dict(custom_weights["mapper"])
    # rec_model.user_embeddings.load_state_dict(custom_weights["user_embeddings"])

    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pairwise Recommendation Model-----\n")

    # '''
    #     Freeze the parameters of the pretrained GPT2 for content model
    # '''
    # # TODO need extension for llama, here only allows user embedding to be trained

    '''
        Define the review pretrain data generator
    '''
    # TODO generate the pretrain data with correct format (pretrain:raw item content, interacted users)

    rec_model.train()

    accelerator.print("-----Pretrain Trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    accelerator.print("\n-----Pretrain Non-trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))


    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    interaction_file_path = os.path.join(dataset_root, 'warm_emb.csv')
    val_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    accelerator.print("-----Preparing Pretrain Dataset-----")

    train_dataset = RecDataset(content_file_path, [interaction_file_path], args.dataset, max_length=args.max_length)
    val_dataset = RecDataset(content_file_path, [val_file_path], args.dataset,
                                       max_length=args.max_length)

    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.pretrain_batch_size,
        per_device_eval_batch_size=args.predict_batch_size,
        warmup_ratio=0.05,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        eval_steps=500,
        save_strategy="epoch",
        output_dir=f"./results/{args.dataset}/{args.LLM_type}",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=None,
        eval_delay=1,
    )

    trainer = Trainer(
        model=rec_model,
        args = training_args,
        data_collator = data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # output = trainer.predict(val_dataset)
    # breakpoint()
    trainer.train(resume_from_checkpoint = None)
    load_path = os.path.join('./model_weight', args.dataset, args.LLM_type, "stage_2")

    #trainer.save_model(load_path)
    rec_model.encoder.save_pretrained(load_path)
    torch.save(
        {
            "mapper": rec_model.mapper.state_dict(),
            "user_embeddings": rec_model.user_embeddings.state_dict(),
        },
        os.path.join(load_path, "custom_weights.pt"),
    )


#TODO set max_length input of LLM to be adjusted
# other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')

# dataset setting
parser.add_argument("--dataset", type=str, default='CiteULike', help="specify the dataset for experiment")

# LLM setting
parser.add_argument('--LLM_root', type=str, default="./LLM_base")
parser.add_argument('--LLM_type', type=str, default="Llama3-1B", help='LLM model type (Llama2-7B, GPT2, Llama3-1B, Llama3-3B,  Llama3-13B)')
parser.add_argument("--max_length", type= int, default=512, help='max taken length of LLM')

# training details
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--pretrain_batch_size', type=int, default=8)
parser.add_argument('--predict_batch_size',  type=int, default=2)

# backbone
parser.add_argument('--backbone_type', type=str, default='LightGCN')
parser.add_argument('--hidden_layers', type=list, default=[512, 256, 512], help='hidden_layers of mapper')

parser.add_argument('--num_epochs', type=int, default=5)

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
