# sentiment_model.py

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from transformers import AdamW

# -------------------------
# 1. Load Dataset
# -------------------------
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# -------------------------
# 2. Load Pre-trained BERT
# -------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Function to tokenize input text
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

print("Tokenizing data...")
train_data = train_data.map(tokenize, batched=True, batch_size=32)
test_data = test_data.map(tokenize, batched=True, batch_size=32)

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)

# -------------------------
# 3. Training Setup
# -------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 4. Train the Model
# -------------------------
print("Training started...")

model.train()
for epoch in range(1):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())

print("Training completed.")

# -------------------------
# 5. Evaluate the Model
# -------------------------
print("Evaluating...")

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print("Accuracy:", accuracy)

# -------------------------
# 6. Save the Model
# -------------------------
model.save_pretrained("saved_sentiment_model")
tokenizer.save_pretrained("saved_sentiment_model")

print("Model saved in folder: saved_sentiment_model")
