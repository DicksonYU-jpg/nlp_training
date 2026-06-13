from omegaconf import OmegaConf
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets impot load_dataset

cfgs = OmegaConf.load("cfg.yml")
sft_args = SFTConfig(**OmegaConf.to_container(cfgs.sft_args))



