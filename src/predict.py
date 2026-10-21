import torch
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME,
    MODEL_PATH,
    DEVICE,
    MAX_LENGTH
)
from src.model import RoBERTa


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = RoBERTa().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return model, tokenizer



def predict_texts(model, tokenizer, texts, device, max_length=256):
    model.eval()
    label_names = ['negative', 'neutral', 'positive']
    results = []

    for text in texts:
        encoding = tokenizer(
            "query: " + str(text),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        input_ids      = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=-1)[0]
            pred   = probs.argmax().item()

        results.append({
            'text':       text,
            'pred_class': pred,
            'pred_label': label_names[pred],
            'probs':      probs.cpu().numpy(),
        })

    return results


def main():
    model, tokenizer = load_model()

    texts = [
        "Отличный сервис, всё быстро и качественно!",
        "Ужасное качество, деньги на ветер.",
        "Товар как товар, ничего особенного."
    ]

    results = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=DEVICE,
        max_length=MAX_LENGTH
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()