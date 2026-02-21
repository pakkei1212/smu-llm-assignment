from abc import ABC, abstractmethod
import json
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    BitsAndBytesConfig,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from src.BaseTrainer import BaseLLMTrainer


class EncoderTrainer(BaseLLMTrainer):
    def __init__(
        self,
        model_name: str,
        labels: list[str],  # Accept list: ["negative", "neutral", "positive"]
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        # 1. Derive mappings from the input list
        self.labels_list = [l.lower() for l in labels]
        self.num_labels = len(self.labels_list)
        self.id2label = {i: label for i, label in enumerate(self.labels_list)}
        self.label2id = {label: i for i, label in enumerate(self.labels_list)}

        super().__init__(model_name, device, load_in_4bit)

    def _load_model(self):
        # use bf16 if available, else fp16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label2id = {v: k for k, v in id2label.items()}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto" if self.load_in_4bit else None,
        )

        # Never allow classifier head to be quantized
        if hasattr(model, "classifier"):
            model.classifier = model.classifier.to(dtype)

        # If not quantized, move full model normally
        if not self.load_in_4bit:
            model.to(self.device)

        return model

    def _task_type(self):
        return TaskType.SEQ_CLS

    def _build_data_collator(self):
        return DefaultDataCollator()

    def load_test_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    @staticmethod
    def _normalize_label(text: str) -> str:
        return (text or "").strip().lower()


    def _label_to_id(self, label: str, labels: list[str]) -> int:
        lab = self._normalize_label(label)
        if lab in labels:
            return labels.index(lab)
        return -1

    # Create a helper method to ensure consistency
    def _tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,            # Must match MAX_LENGTH from training
            padding="max_length",      # Force identical shape
            add_special_tokens=True,   # Ensure [CLS] and [SEP] are added
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_labels(
        self,
        examples: list[dict],
        labels: list[str],
    ) -> np.ndarray:
        """
        Hard predictions using encoder logits.
        """
        model = self.model
        tokenizer = self.tokenizer
        model.eval()

        device = next(model.parameters()).device
        preds = []

        for ex in examples:
            text = ex["input"]

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,        # or self.max_length if defined
                padding=True,
                return_tensors="pt",
            )

            tokens = {k: v.to(device) for k, v in tokens.items()}

            outputs = model(**tokens)
            logits = outputs.logits              # [1, C]
            pred_id = logits.argmax(dim=-1).item()

            preds.append(pred_id)

        return np.array(preds, dtype=int)

    @torch.no_grad()
    def predict_proba(
        self,
        examples: list[dict],
        labels: list[str],
    ) -> np.ndarray:
        """
        Returns probabilities of each class: (N, C)
        """
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        device = next(model.parameters()).device

        all_probs = []
        batch_size = 16

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            texts = [ex["input"] for ex in batch]

            #tokens = self._tokenize(texts).to(device)

            tokens = tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length", 
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits             # [B, C]
                logits = logits.to(torch.float32)
                probs = torch.softmax(logits, dim=-1)

                all_probs.append(probs.cpu().numpy())
            
        return np.vstack(all_probs)

    def evaluate_classification(
        self,
        test_path: str,
        labels: list[str] = None, # Defaults to self.labels_list if None
        average: str = "macro",
        verbose: bool = True,
    ) -> dict:
        """
        Encoder-based evaluation using logits.
        """
        # Use instance labels if none provided to the method
        eval_labels = [l.lower() for l in labels] if labels else self.labels_list
        examples = self.load_test_json(test_path)

        y_true = []
        kept = []

        for ex in examples:
            if "input" not in ex or "output" not in ex:
                continue

            y = self._label_to_id(ex["output"], eval_labels)
            if y == -1:
                continue

            y_true.append(y)
            kept.append(ex)

        if len(kept) == 0:
            raise ValueError("No valid labeled examples found.")

        y_true = np.array(y_true, dtype=int)

        # Predictions
        y_proba = self.predict_proba(kept, labels)
        y_pred = y_proba.argmax(axis=1)

        # print("Min prob:", y_proba.min())
        # print("Max prob:", y_proba.max())
        # print("Row sums (first 5):", y_proba[:5].sum(axis=1))
        # print("Any NaN in proba:", np.isnan(y_proba).any())

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
        except Exception:
            auc = None

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        )
        
        if verbose:
            print("\n[Confusion Matrix]")
            print(cm)

            print("\n[Classification Report]")
            print(report)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc_ovr": float(auc) if auc is not None else None,
        }

    # -------- Save --------
    def save_model(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)

        if self.use_lora:
            # Save LoRA adapter + classifier automatically
            self.model.save_pretrained(path)
        else:
            # Full fine-tuned model
            self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)

    # -------- Load --------
    def load_model(self, path: str, use_lora: bool = False):

        # 1️⃣ Load tokenizer FIRST
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        if use_lora:
            # 2️⃣ Load clean backbone
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )

            # 3️⃣ Attach LoRA adapter (classifier auto-restored)
            self.model = PeftModel.from_pretrained(
                base_model,
                path,
                is_trainable=False,
            )

            print("Loaded LoRA model successfully.")

        else:
            # Full fine-tuned model
            self.model = AutoModelForSequenceClassification.from_pretrained(path)

        self.model.to(self.device)
        self.model.eval()