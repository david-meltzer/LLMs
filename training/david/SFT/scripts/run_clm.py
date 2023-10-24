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

from transformers.trainer_callback import EarlyStoppingCallback

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

def safe_save_model_for_hf_trainer(trainer: Trainer, 
                                   tokenizer: AutoTokenizer, 
                                   output_dir: str):
    """
    Helper method to save model for HF Trainer.

    Args:
        trainer (Trainer): The Hugging Face Trainer object used for training.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        output_dir (str): The directory where the model and tokenizer will be saved.

    Note:
        This function addresses an issue with distributed training and handles the saving process.

    See Also:
        - GitHub Issue: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    """

    # Importing necessary components for distributed training
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    # Get the model from the trainer
    model = trainer.model

    # Define the save policy for FSDP
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # Use FSDP to convert model's state_dict to CPU
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()

    # Save the model and tokenizer if necessary
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(output_dir)



def parse_arge():
    """
    Parse the command line arguments.

    Returns:
        Namespace: Parsed arguments.

    Raises:
        ValueError: If there are conflicting or missing arguments.
    """

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
        default=None,
        help = 'strategy to save model to HF hub')

    parser.add_argument(
        "--output_dir", 
        type=str,
        help = 'output directory to save model.',
    #    default=os.environ["SM_MODEL_DIR"]
    )

    parser.add_argument(
        "--output_data_dir",
        type=str,
        helper = 'output directory to save data.',
    #    default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path used to load dataset."
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="huggingface token. Needed to access Llama model."
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
        help='coefficient for decoupled weight decay.'
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
        help="Set to true to use gradient, or activation, checkpointing.",
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
        '--use_flash_attention',
        type=int,
        default=0,
        help='Set to 1 to use flash-attention'
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        type = int,
        default = 0,
        help ='set to 1 to resume from latest checkpoint.'

    )

    parser.add_argument(
        '--patience',
        type = int,
        default = 5,
        help = 'early stopping patience.'
    )
    
    parser.add_argument(
        '--auto_find_batch_size',
        type = int,
        default = 1,
        help='set to 1 to have trainer automatically find batch size')
    
    parser.add_argument("--fsdp",
                        type=str,
                        default=None,
                        help="Whether to use fsdp.")
    
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Which transformer layer to wrap with fsdp.",
    )
    args, extra = parser.parse_known_args()

    # Log in to the Hugging Face Hub if a token is provided
    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    # Check for conflicting options when loading in 8-bit and 4-bit
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")

    # Check for required arguments when pushing to Hub
    if args.repo_id:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hf_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    # Import necessary modules if using flash attention
    if args.use_flash_attention:
        import utils
        from utils.llama_patch import replace_attn_with_flash_attn
        from utils.llama_patch import forward
        from utils.llama_patch import upcast_layer_for_flash_attention

    return args

class PeftSavingCallback(TrainerCallback):
    """
    Custom callback for saving PEFT models during training.

    Attributes:
        None

    Methods:
        on_save(self, args, state, control, **kwargs):
            Executes custom saving logic for PEFT models.

    """

    def on_save(self, args, state, control, **kwargs):
        """
        Save only the lora layers and remove files for base weights.

        Args:
            args (TrainingArguments): Arguments for configuring the training process.
            state: Current training state.
            control: Control object for the Trainer.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Define the path to save the checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Save the model in the specified format
        kwargs["model"].save_pretrained(checkpoint_path)

        # Remove the pytorch_model.bin file, if present
        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))



# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The neural network model.
        use_4bit (bool, optional): Flag indicating whether 4-bit quantization is used (default is False).

    Returns:
        None
    """
    trainable_params = 0  # Counter for trainable parameters
    all_param = 0  # Counter for all parameters

    # Iterate through named parameters in the model
    for _, param in model.named_parameters():
        num_params = param.numel()  # Get the number of parameters

        # Check for specific conditions related to DS Zero 3 initialization
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params  # Update total parameter count
        if param.requires_grad:  # Check if parameter is trainable
            trainable_params += num_params  # Update trainable parameter count

    # Print the results
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )



# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    """
    Finds and returns the names of all linear modules in the given model.

    Args:
        model (torch.nn.Module): The neural network model.

    Returns:
        list: List of names of Linear modules found in the model.
    """
    lora_module_names = set()  # Set to store unique module names

    # Iterate through named modules in the model
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            # Split the module name based on '.' delimiter
            names = name.split(".")
            # Add the first part (module name) to the set
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove 'lm_head' if present (needed for 16-bit)
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)  # Convert set to list and return

def formatting_prompts_func(example):
    """
    Formats prompts and responses for question/answer format.

    Args:
        example (dict): A dictionary containing 'question' and 'answer' keys.

    Returns:
        list: A list of formatted QA pairs.
    """
    output_texts = []  # List to store formatted QA pairs

    # Iterate through the questions and answers in the example
    for i in range(len(example['question'])):
        # Construct the formatted conversation string
        text = f"### Human: {example['question'][i]}\n ### Assistant: {example['answer'][i]}"
        output_texts.append(text)  # Add the formatted string to the output list

    return output_texts  # Return the list of formatted QA pairs


def sft_collator(tokenizer, response_template = "### Assistant:"):
    """
    Returns a data collator for SFT with a specified response template.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer for processing text input into tokens.
        response_template (str, optional): Template for the response. Default is "### Assistant:".

    Returns:
        DataCollatorForCompletionOnlyLM: Data collator for SFT with the specified response template.
    """
    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

def training_function(args):
    """
    Main function for training a model using the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(args.seed)

    # Use Flash Attention if compatible GPU
    if args.use_flash_attention:
        if torch.cuda.get_device_capability()[0] >= 8:
            from utils.llama_patch import replace_attn_with_flash_attn
            print("using flash attention")
            replace_attn_with_flash_attn()
        else:
            raise ValueError('GPU is not compatible with flash attention. Use Ampere device.')

    # add _qlora to repo_id and run_name if using peft.
    if args.use_peft:
        if args.repo_id: 
            args.repo_id += f'_qlora'
        if args.run_name:
            args.run_name += f'_qlora'

    # Initialize W&B for logging
    if args.report_to_wandb:
        wandb.login(key=args.wandb_token)
        wandb.init(
            job_type='training',
            project=args.project_name,
            entity=args.entity,
            name = args.run_name
        )
        
    print(f'loading from {args.dataset_path}')

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    
    # Set BitsAndBytesConfig for quantization.
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        bnb_config=None

    # load model using bnb_config.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config if args.use_peft else None,
        use_auth_token=args.hf_token
    )
    
    model.config.pretraining_tp = 1
    
    # Identify target modules for quantization.
    if args.use_peft:
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

    # Prepare model for k-bit training (8-bit or 4-bit)
    if args.load_in_8bit or args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    
    # Enable gradient checkpointing, if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_token
    )

    # Set pad_token if None (using a custom new token, see https://github.com/huggingface/transformers/issues/22794
    # since gradients from pad are ignored so if eos_token == pad_token, the model cannot learn to generate EOS)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Determine whether to use bf16 or not
    fp16 = True if not args.bf16 else False
        
    # Define training arguments
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.output_dir) is not None else False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        bf16 = True if args.bf16 else False,
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
        logging_dir=f"{args.output_data_dir}/logs",
        logging_strategy = "steps",
        logging_steps = args.logging_steps,
        save_strategy = 'steps',
        evaluation_strategy = "steps",
        save_steps = .05,
        eval_steps = .05,
        log_level = 'error',
        hub_token = args.hf_token,
        report_to = 'wandb' if args.report_to_wandb else None,
        load_best_model_at_end = args.load_best_model_at_end,
        save_total_limit = 3,
        dataloader_num_workers = 2,
        push_to_hub = True if args.repo_id else False,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.repo_id,
        ddp_timeout=7200,
        auto_find_batch_size = bool(args.auto_find_batch_size)
    )

    # Use Flash Attention if specified
    if args.use_flash_attention:
        from utils.llama_patch import forward    
        from utils.llama_patch import upcast_layer_for_flash_attention
        assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"
        
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        print(f'TORCH DTYPE IS {torch_dtype}')
        model = upcast_layer_for_flash_attention(model, torch_dtype)

    # get PEFT use_peft is true.
    if args.use_peft:
        model = get_peft_model(model, peft_config)
    
    # Define collator for the corresponding tokenizer.
    collator = sft_collator(tokenizer)

    trainer = SFTTrainer(
        model,
        training_args,
        max_seq_length = args.max_seq_length,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        tokenizer = tokenizer,
        formatting_func = formatting_prompts_func,
        packing = False,
        data_collator = collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    # Continue training from the latest checkpoint if specified
    if get_last_checkpoint(args.output_dir) is not None and args.resume_from_checkpoint:
        print("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    # Convert module layers to torch.float32 if not using Flash Attention
    if not args.use_flash_attention:
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

    # Save the trained model
    trainer.save_model(args.output_dir)
    
    # If repo_id is specified, evaluate, create model card, and push to Hub
    if args.repo_id:
        print(f'repo is {args.repo_id}')
        eval_result = trainer.evaluate()
        trainer.create_model_card(model_name=args.repo_id)
        trainer.push_to_hub()
        # Push to


def main():
    args = parse_arge()
    training_function(args)

if __name__ == "__main__":
    main()   