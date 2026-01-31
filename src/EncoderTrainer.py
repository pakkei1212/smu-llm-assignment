from abc import ABC, abstractmethod
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from src.BaseTrainer import BaseLLMTrainer


class EncoderTrainer(BaseLLMTrainer):
    def _load_model(self):
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        return model

    def _task_type(self):
        return TaskType.FEATURE_EXTRACTION

    def _build_data_collator(self):
        return DefaultDataCollator()
