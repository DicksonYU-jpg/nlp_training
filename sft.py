from pathlib import Path

import wandb
from omegaconf import OmegaConf
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from utils.resume import find_latest_checkpoint


def sft_train(cfg_path: str):    
    cfgs = OmegaConf.load(cfg_path)
    wandb.init(project=cfgs.wandb_args.project, name=cfgs.wandb_args.name)
    
    print(f"Base Model: {cfgs.model}")
    print(f"Dataset: {cfgs.dataset}")
    
    resume_from = find_latest_checkpoint(cfgs.sft_args.output_dir)
    if resume_from:
        print(f"Resuming training from: {resume_from}")
    else:
        print("No checkpoint found — starting fresh.")
    
    sft_args = SFTConfig(**OmegaConf.to_container(cfgs.sft_args))
    peft_args = LoraConfig(**OmegaConf.to_container(cfgs.lora_args))

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model)
    model = AutoModelForCausalLM.from_pretrained(cfgs.model, dtype="auto")
    
    # Load dataset
    dataset = load_dataset(cfgs.dataset)
    train, eval = dataset.train_test_split(test_size=0.05)
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        args = sft_args,
        peft_config = peft_args,
        train_dataset = train,
        eval_dataset = eval
    )
    
    trainer.train(resume_from_checkpoint=resume_from)

def main():
    # get the full path of cfgs
    cfg_paths = [ path for path in Path("./cfgs").glob("*.yml") if path.is_file() ]
    
    for cfg_path in cfg_paths:
        print(f"Starting SFT training with config: {cfg_path}")
        sft_train(str(cfg_path))
        print(f"Finished SFT training with config: {cfg_path}\n\n")

if __name__ == "__main__":
    main()
    
    








