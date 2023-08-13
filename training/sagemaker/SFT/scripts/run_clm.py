import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
from peft.tuners.lora import LoraLayer
import wandb


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )
    parser.add_argument(
        "--hf_token", type=str, default=HfFolder.get_token(), help="hf token."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help='number of gradient accumulation steps.'
    )

    parser.add_argument(
        "wandb_token",
        type=str,
        help='token for wandb'
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help='max size of input sequence.'
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help='max number of training steps.'
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help='Log results every n-steps'
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size to use for validation.",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default='paged_adamw_32bit',
        help="optimizer to use for training.",
    )

    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )

    parser.add_argument(
        "--warmup_ratio",
        type = float,
        default=.03,
        help='fraction of run dedicated to warmup'
    )

    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=True,
        help='set to true to group batches by length'
    )

    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default='SFT_training_dm',
        help='name of wandb project',
    )

    parser.add_argument(
        "--entity",
        type=str,
        default='ft-llmmm',
        help='name of wandb team name.'
    )

    parser.add_argument(
        "--run_name",
        type=str,
        help='name of wandb run.'
    )

    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def sft_collator(tokenizer, response_template = " ### Answer:"):
    
    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

def training_function(args):
    # set seed
    set_seed(args.seed)

    if args.wandb_token:
        wandb.login(key=args.wandb_token)
        
        wandb.init(
            job_type='training',
            project=args.project_name,
            entity=args.entity,
            name = args.run_name
            )

    dataset = load_from_disk(args.dataset_path)
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,  # this is needed for gradient checkpointing
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=args.hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # create peft config
    model = create_peft_model(
        model, 
        gradient_checkpointing=args.gradient_checkpointing, 
        bf16=args.bf16
    )

    if not args.bf16:
        fp16 = True
    else:
        fp16 = False

    # Define training args
    output_dir = "./tmp/llama2"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        fp16=fp16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps = args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        group_by_length=args.group_by_length,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        #logging_strategy="steps",
        #logging_steps=args.logging_steps,
        #save_strategy="epoch",
        log_level = 'error',
        hub_token=args.hf_token,
        report_to='wandb' if args.wandb_token else None
        #max_grad_norm=0.3
    )

    #collator=sft_collator(tokenizer)

    trainer = SFTTrainer(
        model,
        training_args,
        max_seq_length = args.max_seq_length,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        #dataset_text_field='QA',
        packing=False,
        #data_collator=collator
        )

    # Create Trainer instance
    #trainer = Trainer(
    #    model=model,
    #    args=training_args,
    #    train_dataset=dataset,
    #    data_collator=default_data_collator,
    #)

    # Start training
    trainer.train()

    sagemaker_save_dir="/opt/ml/model/"
    
    if args.merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_auth_token=args.hf_token
        )  
        # Merge LoRA and base model and save
        model = model.merge_and_unload()        
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True
        )

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              use_auth_token=args.hf_token)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
