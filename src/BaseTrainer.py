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


def resolve_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cuda", "cpu"}:
        return device
    raise ValueError("device must be 'auto', 'cuda', or 'cpu'")


class BaseLLMTrainer(ABC):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model()
        self.task_type = self._task_type()
        self.data_collator = self._build_data_collator()

    # -------- Abstract hooks --------
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _task_type(self) -> TaskType:
        pass

    @abstractmethod
    def _build_data_collator(self):
        pass

    # -------- LoRA --------
    def configure_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules=None,
        bias: str = "none",
    ):
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=self.task_type,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # -------- Training --------
    def train(
        self,
        train_dataset,
        eval_dataset,
        training_args: TrainingArguments,
        compute_metrics=None,
    ):
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.trainer = trainer

    # -------- Save / Load --------
    def save_lora_adapters(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_lora_adapters(self, path: str):
        self.model = PeftModel.from_pretrained(self.model, path)
