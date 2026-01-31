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


class DecoderTrainer(BaseLLMTrainer):
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=self.load_in_4bit,
            torch_dtype=torch.float32,
        )
        if not self.load_in_4bit:
            model = model.to(self.device)
        return model

    def _task_type(self):
        return TaskType.CAUSAL_LM

    def _build_data_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def generate(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
