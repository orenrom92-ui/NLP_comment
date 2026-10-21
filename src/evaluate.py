import torch
import numpy as np
import torch.utils.data as data

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME,
    MODEL_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    DEVICE,
)
from src.datasets import TextDataset, transform_numpy
from src.model import RoBERTa
from src.utils import func_metrics_multiclass


def evaluate_model():

    dataset = load_dataset("ai-forever/ru-reviews-classification")

    x_test, y_test = transform_numpy(dataset, 'test')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_dataset = TextDataset(
        x_test,
        y_test,
        tokenizer,
        MAX_LENGTH
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_model = RoBERTa().to(DEVICE)

    test_model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    test_model.eval()

    test_predictions = []
    test_labels = []

    with torch.no_grad():
        test_tqdm = tqdm(test_loader, desc='Test', leave=False)

        for token_emb, token_mask, label in test_tqdm:
            x = token_emb.to(DEVICE, non_blocking=True)
            m = token_mask.to(DEVICE, non_blocking=True)
            y = label.to(DEVICE, non_blocking=True).long()

            logits = test_model(x, m)
            preds = logits.argmax(dim=-1)

            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)

    test_metrics = func_metrics_multiclass(
        test_predictions,
        test_labels
    )

    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1']:.4f}")
    print(f"Test precision:  {test_metrics['precision']:.4f}")
    print(f"Test recall:     {test_metrics['recall']:.4f}")

    return test_metrics


if __name__ == "__main__":
    evaluate_model()