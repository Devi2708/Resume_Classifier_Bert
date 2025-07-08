from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("csv", data_files="data/labeled_data.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset["train"].train_test_split(test_size=0.2)

train_ds, test_ds = dataset["train"], dataset["test"]
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

args = TrainingArguments(output_dir="models", evaluation_strategy="epoch", per_device_train_batch_size=4, num_train_epochs=2, logging_dir="logs")

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
trainer.train()

model.save_pretrained("models/resume_classifier")
tokenizer.save_pretrained("models/resume_classifier")