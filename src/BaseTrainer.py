from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

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

from src.ClassificationEvalTrainer import ClassificationEvalTrainer

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
        max_length = 128
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # DeBERTa-specific fix: Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = self._load_model()
        self.class_weights = None
        self.task_type = self._task_type()
        self.data_collator = self._build_data_collator()
        
        self.use_lora = False
        self.max_length = max_length

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
        target_modules: list[str] | None = None,
        bias: str = "none",
    ):
        """
        Configure LoRA adapters for encoder-based models.

        Defaults are chosen to be:
        - encoder-safe
        - stable under BF16
        - suitable for hyperparameter tuning
        """
            
        if target_modules is None:
            target_modules = ["query", "value"]

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

        self.use_lora = True 

    # -------- Training --------
    def train(
        self,
        train_dataset,
        eval_dataset,
        training_args: TrainingArguments,
        data_collator: DataCollatorForLanguageModeling,
        classification_eval_fn=None,
    ):
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        trainer = ClassificationEvalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            class_weights=self.class_weights,
            classification_eval_fn=classification_eval_fn,
        )

        trainer.train()
        self.trainer = trainer

        if classification_eval_fn is not None:
            return classification_eval_fn()
        else:
            return trainer.evaluate()

    '''# -------- Save / Load --------
    def save_lora_adapters(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_lora_adapters(self, path: str):
        self.model = PeftModel.from_pretrained(
            self.model,
            path,
            is_trainable=False
        )
        self.model.eval()'''

    # -------- Save --------
    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # -------- Load --------
    def load_model(self, path: str, use_lora: bool = False):
        if use_lora:
            self.model = PeftModel.from_pretrained(
                self.model,
                path,
                is_trainable=False
            )
        else:
            self.model = type(self.model).from_pretrained(path)

        self.model.eval()
