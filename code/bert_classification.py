from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define seed value
SEED = 42

# Set seed for Python's random module
random.seed(SEED)

# Set seed for NumPy
np.random.seed(SEED)

# Set seed for PyTorch
torch.manual_seed(SEED)

# If using a GPU
torch.cuda.manual_seed_all(SEED)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=2) 
model.to(device)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
		

with open('turkish_liar.json') as file:
    data = json.load(file)

sentences, labels = [], []
for item in data.keys():
    sentences.extend(data[item]['claude_correct'])
    labels.extend([1]*len(data[item]['claude_correct']))
    
    sentences.extend(data[item]['claude_incorrect'])
    labels.extend([0]*len(data[item]['claude_incorrect']))
    
    sentences.extend(data[item]['gpt_correct'])
    labels.extend([1]*len(data[item]['gpt_correct']))
    
    sentences.extend(data[item]['gpt_incorrect'])
    labels.extend([0]*len(data[item]['gpt_incorrect']))
	
# Split into train (80%) and temp (20%)
sentences_train, sentences_temp, labels_train, labels_temp = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Further split temp into dev (10%) and test (10%)
sentences_dev, sentences_test, labels_dev, labels_test = train_test_split(
    sentences_temp, labels_temp, test_size=0.5, random_state=42
)
	
# Create datasets
train_dataset = CustomDataset(sentences_train, labels_train, tokenizer, max_len=128)
dev_dataset = CustomDataset(sentences_dev, labels_dev, tokenizer, max_len=128)
test_dataset = CustomDataset(sentences_test, labels_test, tokenizer, max_len=128)

# Create DataLoaders
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_data_loader = DataLoader(dev_dataset, batch_size=16)
test_data_loader = DataLoader(test_dataset, batch_size=16)





optimizer = AdamW(model.parameters(), lr=2e-5)

# Early stopping parameters
best_val_loss = np.inf
patience = 3  # Number of epochs to wait before stopping
epochs_without_improvement = 0

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    val_loss /= len(val_data_loader)

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_data_loader)}, Validation Loss: {val_loss}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered")
        break
		
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.eval()  # Set to evaluation mode

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(model, data_loader):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels
	
# Evaluate on test set
test_predictions, test_labels = evaluate(model, test_data_loader)

# Calculate metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
