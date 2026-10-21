import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import (
    MODEL_NAME,
    MODEL_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    NUM_EPOCHS,
    RANDOM_SEED,
    DEVICE,
)
from src.datasets import TextDataset, transform_numpy
from src.model import RoBERTa
from src.utils import conf_seed, func_metrics_multiclass


def train_model(model, train_loader, val_loader, num_epochs):

    optimizer = torch.optim.AdamW(params = model.parameters(), lr=2e-5, weight_decay = 0.01)
    loss_func = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    train_losses = []
    val_losses = []
    best_val_f1 = 0

    for i in range(num_epochs):
        print(f'\nEpoch {i + 1}/{num_epochs}')
        model.train()
        loss_mean_train = 0
        lm_count_train = 0

        train_tqdm = tqdm(train_loader, leave=True)
        for token_emb, token_mask, lable in train_tqdm:
            x = token_emb.to(DEVICE, non_blocking=True)
            m = token_mask.to(DEVICE, non_blocking=True)
            y = lable.to(DEVICE, non_blocking=True).long()

            predict = model(x, m)
            loss = loss_func(predict, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            lm_count_train += 1
            loss_mean_train = 1/lm_count_train * loss.item() + (1 - 1/lm_count_train) * loss_mean_train
            train_tqdm.set_description(f"Epoch [{i+1}/{num_epochs}], loss_mean={loss_mean_train:.3f}")

        train_losses.append(loss_mean_train)

        model.eval()
        loss_mean_val = 0
        lm_count_val = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            val_tqdm = tqdm(val_loader, leave=True)

            for token_emb, token_mask, lable in val_tqdm:
                x = token_emb.to(DEVICE, non_blocking=True)
                m = token_mask.to(DEVICE, non_blocking=True)
                y = lable.to(DEVICE, non_blocking=True).long()

                predict = model(x, m)
                loss = loss_func(predict, y)

                lm_count_val += 1
                loss_mean_val = 1/lm_count_val * loss.item() + (1 - 1/lm_count_val) * loss_mean_val
                val_tqdm.set_description(f"Epoch [{i+1}/{num_epochs}], loss_mean={loss_mean_val:.3f}")

                predictions = torch.argmax(predict, dim=-1).cpu().numpy()
                val_predictions.extend(predictions)
                val_labels.extend(y.cpu().numpy())

        val_losses.append(loss_mean_val)

        metrics = func_metrics_multiclass(np.array(val_predictions), np.array(val_labels))
        val_f1 = metrics['f1']
        val_acc = metrics['accuracy']

        print(f'Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val accuracy: {val_acc:.3f} | Val F1: {val_f1:.3f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f'\nTraining Complete. Best validation F1: {best_val_f1:.4f}')

    return model, train_losses, val_losses


def prepare_data():
    dataset = load_dataset("ai-forever/ru-reviews-classification")

    x_train, y_train = transform_numpy(dataset, 'train')
    x_valid, y_valid = transform_numpy(dataset, 'validation')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextDataset(
        x_train,
        y_train,
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = TextDataset(
        x_valid,
        y_valid,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader

def main():
    conf_seed(RANDOM_SEED)

    train_loader, val_loader = prepare_data()

    model = RoBERTa()
    model = model.to(DEVICE)

    final_model, train_l, val_l = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
    )

    return final_model, train_l, val_l


if __name__ == "__main__":
    main()