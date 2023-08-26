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
    TrainerCallback,
)
import evaluate
import pandas as pd

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
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
import utils
from utils.llama_patch import replace_attn_with_flash_attn
from utils.llama_patch import forward
from utils.llama_patch import upcast_layer_for_flash_attention

def safe_save_model_for_hf_trainer(trainer: Trainer, tokenizer: AutoTokenizer, output_dir: str):
    """Helper method to save model for HF Trainer."""
    # see: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    model = trainer.model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(output_dir)


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
        '--repo_id',
        type=str,
        help='name of huggingface repo to push model to.'
    )

    parser.add_argument(
        "--hub_strategy",
        type=str,
        default=None)

    parser.add_argument(
        "--output_dir", 
        type=str,
        #default=os.environ["SM_MODEL_DIR"]
    )

    parser.add_argument(
        "--output_data_dir",
        type=str,
        #default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset."
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="hf token."
    )

    parser.add_argument(
        '--report_to_wandb',
        type=int,
        default=1,
        help='whether to log to wandb')

    parser.add_argument(
        "--wandb_token",
        type=str,
        help='token for wandb'
    )

    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train for."
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default = -1,
        help='max number of training steps.'
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size to use for training.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size to use for validation.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help='number of gradient accumulation steps.'
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help='max size of input sequence.'
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help='Log results every n-steps'
    )

    parser.add_argument(
        "--optim",
        type=str,
        default='adamw_torch_fused',
        help="optimizer to use for training.",
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5, 
        help="Learning rate to use for training."
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help='rank for LORA adapter weights.'
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help='alpha for LORA adapter weights.'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help='dropout in lora weights.'
    )

    parser.add_argument(
        "--load_in_4bit",
        type=int,
        default=1,
        help='Whether to load model in 4-bits.'
    )

    parser.add_argument(
        "--load_in_8bit",
        type=int,
        default=0,
        help='Whether to load model in 8-bits.'
    )

    parser.add_argument(
        "--use_peft",
        type=int,
        default=1,
        help='Set to true to use parameter efficient fine-tuning.'
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        help="Path to deepspeed config file.",
    )

    parser.add_argument(
        "--bf16",
        type=int,
        default=1 if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    
    parser.add_argument(
        "--group_by_length",
        type=int,
        default=1,
        help='set to true to group batches by length'
    )
    
    parser.add_argument(
        "--merge_weights",
        type=int,
        default=0,
        help="Whether to merge LoRA weights with base model.",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Seed to use for training."
    )

    parser.add_argument(
        "--warmup_ratio",
        type = float,
        default=.03,
        help='fraction of run dedicated to warmup'
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

    parser.add_argument(
        '--load_best_model_at_end',
        type=int,
        help='Set to 1 to load best model at end of training. Else set to 0.',
        default=1
    )

    parser.add_argument(
        '--use_sagemaker',
        type=int,
        default=1,
        help='Set to 1 if using sagemaker. Else set to 0.'
    )
    
    parser.add_argument(
        '--torch_compile',
        type=int,
        default=0,
        help='Set to 1 to compile model. Else set to 0.'
    )
    parser.add_argument(
        '--use_flash_attention',
        type=int,
        default=0,
        help='Set to 1 to use flash-attention'
    )
    
#    parser.add_argument("--fsdp",
#                        type=str,
#                        default=None,
#                        help="Whether to use fsdp.")
#    
#    parser.add_argument(
#        "--fsdp_transformer_layer_cls_to_wrap",
#        type=str,
#        default=None,
#        help="Which transformer layer to wrap with fsdp.",
#    )
    
    #parser.add_argument(
    #    '--metric_for_best_model',
    #    type=str,
    #    default='bertscore_f1',
    #    help='metric on evaluation set used to choose best model'
    #)
    
    args, extra = parser.parse_known_args()
    
    print(f'args is {args}')
    print(f'extra is {extra}')

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)
    
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        
    if args.repo_id:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hf_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    #if args.use_sagemaker:
    #    args.output_dir = '/tmp'
    
    if args.torch_compile:
        print('COMPILE SET TO TRUE')

    return args

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


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


#def create_peft_model(model,args, bf16=True):
#    
#    # prepare int-4 model for training
#    model = prepare_model_for_kbit_training(
#        model, use_gradient_checkpointing=args.gradient_checkpointing
#    )
#    if args.gradient_checkpointing:
#        model.gradient_checkpointing_enable()
#
#    # get lora target modules
#    modules = find_all_linear_names(model)
#    print(f"Found {len(modules)} modules to quantize: {modules}")
#
#    peft_config = LoraConfig(
#        r=args.lora_r,
#        lora_alpha=args.lora_alpha,
#        target_modules=modules,
#        lora_dropout=args.lora_dropout,
#        bias="none",
#        task_type=TaskType.CAUSAL_LM,
#    )
#
#    model = get_peft_model(model, peft_config)
#
#    if not args.use_flash_attention:
#    # pre-process the model by upcasting the layer norms in float 32 for
#        for name, module in model.named_modules():
#            if isinstance(module, LoraLayer):
#                if bf16:
#                    module = module.to(torch.bfloat16)
#            if "norm" in name:
#                module = module.to(torch.float32)
#            if "lm_head" in name or "embed_tokens" in name:
#                if hasattr(module, "weight"):
#                    if bf16 and module.weight.dtype == torch.float32:
#                        module = module.to(torch.bfloat16)
#    else:
#        
#
#    model.print_trainable_parameters()
#    
#    return model

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Human: {example['question'][i]}\n ### Assistant: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def sft_collator(tokenizer, response_template = "### Assistant:"):
    
    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

#def compute_metrics(eval_pred):
#    
#    rouge=evaluate.load('rouge')
#    bertscore=evaluate.load('bertscore')
#    
#    predictions,labels=eval_pred
#    decoded_preds=self.tokenizer.batch_decode(predictions)
#    labels=np.where(labels!=-100,labels,self.tokenizer.pad_token_id)
#    decoded_labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
#
#    result={}
#
#    result['bertscore'] = bertscore.compute(predictions=decoded_preds,
#                                        references=decoded_labels,
#                                        lang='en',
#                                            model_type="distilbert-base-uncased")
#
#    result['rouge'] = rouge.compute(predictions=decoded_preds,
#                            references=decoded_labels)
#
#    output={}
#    for k in result:
#        for met in result[k]:
#            if met!='hashcode':
#                output[k+'_'+met]=np.mean(result[k][met])
#
#    return output

def training_function(args):
    # set seed
    set_seed(args.seed)

    if args.use_flash_attention:
        if torch.cuda.get_device_capability()[0] >= 8:
            #from utils.llama_patch import replace_attn_with_flash_attn
            print("using flash attention")
            replace_attn_with_flash_attn()
        else:
            raise ValueError('GPU is not compatible with flash attention. Use Ampere device.')
    
    if args.use_peft:
        if args.repo_id: 
            args.repo_id += f'_r_{args.lora_r}_alpha_{args.lora_alpha}'
        if args.run_name:
            args.run_name += f'_r_{args.lora_r}_alpha_{args.lora_alpha}'

    if args.report_to_wandb:
        wandb.login(key=args.wandb_token)
        
        wandb.init(
            job_type='training',
            project=args.project_name,
            entity=args.entity,
            name = args.run_name
            )
        
    print(f'loading from {args.dataset_path}')

    dataset = load_from_disk(args.dataset_path)
    # load model from the hub with a bnb config
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
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
    
    model.config.pretraining_tp = 1
    
    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
       
    if not args.bf16:
        fp16 = True
    else:
        fp16 = False
        
    # Define training args

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.output_dir) is not None else False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        bf16 = True if args.bf16 else False,  # Use BF16 if available
        fp16 = fp16,
        learning_rate = args.lr,
        num_train_epochs = args.epochs,
        max_steps = args.max_steps,
        gradient_checkpointing = args.gradient_checkpointing,
        optim = args.optim,
        warmup_ratio = args.warmup_ratio,
        weight_decay = args.weight_decay,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        group_by_length = args.group_by_length,
        # logging strategies
        logging_dir=f"{args.output_data_dir}/logs",
        logging_strategy = "steps",
        logging_steps = args.logging_steps,
        save_strategy = 'steps',
        evaluation_strategy = "steps",
        save_steps = .1,
        eval_steps = .1,
        log_level = 'error',
        hub_token = args.hf_token,
        report_to = 'wandb' if args.report_to_wandb else None,
        load_best_model_at_end = args.load_best_model_at_end,
        save_total_limit = 3,
        dataloader_num_workers = 2,
        push_to_hub = True if args.repo_id else False,
        hub_strategy=args.hub_strategy,
        #max_grad_norm=0.3,
        hub_model_id=args.repo_id,
        torch_compile=args.torch_compile,
        #metric_for_best_model=args.metric_for_best_model,
        ddp_timeout=7200,
#        fsdp=args.fsdp,
#        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
    )

    if args.use_flash_attention:
        #from .utils.llama_patch import forward    
        assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"
        
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        print(f'TORCH DTYPE IS {torch_dtype}')
        model = upcast_layer_for_flash_attention(model, torch_dtype)

    if args.use_peft:
        model = get_peft_model(model, peft_config)
    
    collator=sft_collator(tokenizer)
    
    #if args.torch_compile:
    #    model = torch.compile(model)
    

    trainer = SFTTrainer(
        model,
        training_args,
        max_seq_length = args.max_seq_length,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        tokenizer = tokenizer,
        formatting_func = formatting_prompts_func,
        #dataset_text_field='QA',
        packing = False,
        data_collator = collator,
        #compute_metrics = compute_metrics
        )
    
    original_performance = trainer.evaluate()
    wandb.log({'initial-performance': wandb.Table(dataframe=pd.DataFrame(original_performance, index=["Performance"]))})

    trainer.train()

    if args.repo_id:
        print(f'repo is {args.repo_id}')
        eval_result = trainer.evaluate()
        trainer.create_model_card(model_name=args.repo_id)
        trainer.push_to_hub()
        
    final_performance = trainer.evaluate()
    wandb.log({'final-performance': wandb.Table(dataframe=pd.DataFrame(final_performance, index=["Performance"]))})
    
    trainer.save_model(args.output_dir)

def main():
    args = parse_arge()
    
    print(args)
    training_function(args)


if __name__ == "__main__":
    main()

#        model.push_to_hub(args.repo_id)
#        tokenizer.push_to_hub(args.repo_id)
    

#    if args.use_sagemaker:
#        save_dir="/opt/ml/model/"
#
#        if args.merge_weights:
#            # merge adapter weights with base model and save
#            # save int 4 model
#            trainer.model.save_pretrained(args.output_dir,
#                                          safe_serialization=False)
#            # clear memory
#            del model
#            del trainer
#            torch.cuda.empty_cache()
#
##            from peft import AutoPeftModelForCausalLM
##
#            # load PEFT model in fp16
#            model = AutoPeftModelForCausalLM.from_pretrained(
#                args.output_dir,
#                low_cpu_mem_usage=True,
#                torch_dtype=torch.float16,
#                use_auth_token=args.hf_token
#            )  
#            # Merge LoRA and base model and save
#            model = model.merge_and_unload()        
#            model.save_pretrained(
#                save_dir, safe_serialization=True, max_shard_size="2GB"
#            )
#        else:
#            trainer.model.save_pretrained(
#                save_dir, safe_serialization=True
#            )
#
#
        # save tokenizer for easy inference
        #tokenizer = AutoTokenizer.from_pretrained(args.model_id,
        #                                          use_auth_token=args.hf_token)
        #tokenizer.save_pretrained(save_dir)