from abc import ABC, abstractmethod
import json
import re
import torch
import numpy as np

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
    BitsAndBytesConfig,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from src.BaseTrainer import BaseLLMTrainer


class DecoderTrainer(BaseLLMTrainer):
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
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto" if self.load_in_4bit else None,
        )

        if not self.load_in_4bit:
            model.to(self.device)

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

    # =========================================================
    # ✅ Added: Evaluation helpers for classification
    # =========================================================

    def build_prompt(self, instruction: str, input_text: str) -> str:
        """Matches your training style."""
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    def load_test_json(self, path: str):
        """
        Loads a JSON file
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    @staticmethod
    def _normalize_label(text: str) -> str:
        text = (text or "").strip().lower()
        # Take first token/word; your labels are single words like positive/negative/neutral
        text = re.split(r"[\s\n\r\t,;:.]+", text)[0]
        return text

    def _label_to_id(self, label: str, labels: list[str]) -> int:
        lab = self._normalize_label(label)
        if lab in labels:
            return labels.index(lab)
        return -1

    def predict_labels(
        self,
        examples: list[dict],
        labels: list[str],
        max_new_tokens: int = 6
    ) -> np.ndarray:
        """
        Hard predictions by generating text and mapping to label ids.
        """

        preds = []
        for ex in examples:
            prompt = self.build_prompt(ex["instruction"], ex["input"])
            gen = self.generate(prompt, max_new_tokens=max_new_tokens)
            pred_text = gen[len(prompt):].strip()
            pred_id = self._label_to_id(pred_text, labels)

            # If unexpected output → keep as -1 (counted as wrong)
            preds.append(pred_id)

        return np.array(preds, dtype=int)

    @torch.no_grad()
    def _label_sequence_logprob(self, prompt: str, label: str, max_length: int = 2048) -> float:
        """
        Compute log P(label | prompt) for decoder-only LM by scoring the label tokens
        appended after the prompt. Works even if label is multi-token.
        """
        # Tokenize prompt alone
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Tokenize label alone (prepend a space to help tokenization for some tokenizers)
        label_ids = self.tokenizer(" " + label, add_special_tokens=False)["input_ids"]

        # Build full input
        input_ids = prompt_ids + label_ids
        if len(input_ids) > max_length:
            # truncate from the left (keep the end which includes response)
            input_ids = input_ids[-max_length:]
            # recompute prompt length after truncation
            # (best effort; if truncation cuts prompt, label start shifts)
            # We'll approximate by ensuring label_ids still at the end:
            label_len = len(label_ids)
            prompt_len = max(0, len(input_ids) - label_len)
        else:
            prompt_len = len(prompt_ids)
            label_len = len(label_ids)

        input_tensor = torch.tensor([input_ids], device=self.model.device)
        attn = torch.ones_like(input_tensor)

        out = self.model(input_ids=input_tensor, attention_mask=attn)
        logits = out.logits  # [1, T, V]

        # For token t, model predicts input_ids[t] at logits[t-1]
        # We need logprobs of the label tokens positions only.
        start = prompt_len
        end = prompt_len + label_len

        # label token positions are [start, end-1]
        # predicted by logits at positions [start-1, end-2]
        logprobs_sum = 0.0
        for pos in range(start, end):
            prev_pos = pos - 1
            if prev_pos < 0 or pos >= logits.size(1):
                continue
            token_id = input_ids[pos]
            lp = torch.log_softmax(logits[0, prev_pos, :], dim=-1)[token_id]
            logprobs_sum += float(lp.detach().cpu())

        return logprobs_sum

    def predict_proba(
        self,
        examples: list[dict],
        labels: list[str],
        max_length: int = 2048
    ) -> np.ndarray:
        """
        Returns probabilities of each class shape: (N, C)
        by scoring each label with sequence log-prob and softmaxing across labels.
        """

        all_probs = []
        for ex in examples:
            prompt = self.build_prompt(ex["instruction"], ex["input"])

            scores = []
            for lab in labels:
                s = self._label_sequence_logprob(prompt, lab, max_length=max_length)
                scores.append(s)

            scores = np.array(scores, dtype=np.float64)
            # softmax in numpy (stable)
            scores = scores - scores.max()
            probs = np.exp(scores) / np.exp(scores).sum()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def evaluate_classification(
        self,
        test_path: str,
        labels: list[str] = ("negative", "neutral", "positive"),
        average: str = "macro",
        max_new_tokens: int = 6
    ) -> dict:
        """
        Evaluates accuracy/precision/recall/F1 and AUC on test.json.

        - Uses generation for hard predictions (y_pred)
        - Uses label-sequence scoring for probabilities (y_proba) to compute AUC
        """
        labels = [l.lower() for l in list(labels)]
        examples = self.load_test_json(test_path)

        # Ground truth
        y_true = []
        kept = []
        for ex in examples:
            if "instruction" not in ex or "input" not in ex or "output" not in ex:
                continue
            y = self._label_to_id(ex["output"], labels)
            if y == -1:
                continue
            y_true.append(y)
            kept.append(ex)

        if len(kept) == 0:
            raise ValueError("No valid labeled examples found in test.json.")

        y_true = np.array(y_true, dtype=int)

        # Predictions
        y_pred = self.predict_labels(kept, labels=labels, max_new_tokens=max_new_tokens)

        # Probabilities for AUC
        y_proba = self.predict_proba(kept, labels=labels)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        # Multiclass AUC (one-vs-rest)
        # Works when y_proba is (N, C)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
        except Exception:
            auc = None  # if sklearn can't compute due to edge cases

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc_ovr": auc,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        return metrics