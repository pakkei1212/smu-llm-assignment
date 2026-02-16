from transformers import Trainer
import torch.nn.functional as F

class ClassificationEvalTrainer(Trainer):
    """
    Hugging Face Trainer extended to support
    label-token-based classification metrics.
    """

    def __init__(self, class_weights=None, classification_eval_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.classification_eval_fn = classification_eval_fn
        

    def evaluate(self, *args, **kwargs):
        # Standard HF evaluation (loss, ppl, etc.)
        metrics = super().evaluate(*args, **kwargs)

        # Inject custom classification metrics
        if self.classification_eval_fn is not None:
            cls_metrics = self.classification_eval_fn()

            logged = {}
            for k, v in cls_metrics.items():
                if k != "confusion_matrix":
                    metrics[f"eval_{k}"] = v
                    logged[f"eval_{k}"] = v

            print("\n[Classification Metrics]")
            for k, v in cls_metrics.items():
                if k != "confusion_matrix":
                    if isinstance(v, (int, float)):
                        print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")

            self.log(logged)

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if getattr(self, "class_weights", None) is not None:
            #print("Using class weights for loss computation.")
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights.to(logits.device),
            )
        else:
            #print("Using standard loss computation.")
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss