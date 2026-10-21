import torch
import torch.utils.data as data


def transform_numpy(dataset, name):
    df = dataset[name].to_pandas()
    x = df['text'].to_numpy()
    y = df['label'].to_numpy()
    return x, y


class TextDataset(data.Dataset):
    def __init__(self, texts, labels, tokenizer, MAX_LEN):
        self.MAX_LEN = MAX_LEN
        self.tokenizers = tokenizer
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.text = texts

    def __getitem__(self, idx):
        x = "query: " + self.text[idx]
        y = self.labels[idx]

        encoding = self.tokenizers(
            x,
            truncation=True,
            padding='max_length',
            max_length=self.MAX_LEN,
            return_tensors='pt'
        )

        return (
            encoding['input_ids'].flatten(),
            encoding['attention_mask'].flatten(),
            self.labels[idx]
        )

    def __len__(self):
        return len(self.labels)