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


class Seq2SeqTrainer(BaseLLMTrainer):
    def _load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            load_in_4bit=self.load_in_4bit,
            torch_dtype=torch.float32,
        )
        if not self.load_in_4bit:
            model = model.to(self.device)
        return model

    def _task_type(self):
        return TaskType.SEQ_2_SEQ_LM

    def _build_data_collator(self):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

    def generate(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
