import os
import torch
from transformers import AutoTokenizer, TrainingArguments
from transformers import LlamaForCausalLM
from accelerate import Accelerator
def setup_environment(args):
    torch.cuda.manual_seed_all(args.seed)
    accelerator = Accelerator(cpu=False)
    if torch.cuda.is_available():
        print(f"device: {accelerator.device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")
    return accelerator

def get_tokenizer(args):
    llm_root = resolve_llm_root(args)
    tokenizer = AutoTokenizer.from_pretrained(llm_root, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_llama_model(args):
    llm_root = resolve_llm_root(args)
    model = LlamaForCausalLM.from_pretrained(llm_root, attn_implementation="eager")
    config = model.config
    # config.hidden_layers = args.hidden_layers
    return model, config

def get_user_embeddings(num_users, hidden_size):
    return torch.nn.Embedding(num_users, hidden_size)

def resolve_llm_root(args):
    llm_type = args.LLM_type
    if llm_type == 'LLama2':
        return os.path.join(args.LLM_root, args.LLM_type)
    elif llm_type == 'Llama3-1B':
        return '/home/models/Llama-3.2-1B'
    elif llm_type == "Llama3-3B":
        return '/home/models/Llama-3.2-3B'
    elif llm_type == "Llama3-8B":
        return '/root/autodl-tmp/model'
    elif llm_type == "Llama3-13B":
        return '/home/models/Llama-2-13b-hf'
    else:
        raise ValueError("Unsupported LLM_type provided.")

def get_dpo_training_cofig(args):
    return TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.dpo_batch_size,
        per_device_eval_batch_size=args.predict_batch_size,
        warmup_ratio=0.05,
        num_train_epochs=args.dpo_num_epochs,
        learning_rate=args.dpo_lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        eval_steps=500,
        save_strategy="epoch",
        output_dir=f"./results/{args.dataset}/{args.LLM_type}",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=None,
        max_grad_norm=1.0,
        eval_delay=1,
        logging_steps=10,
        logging_strategy="steps"
    )