import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from logRegressionModel import LogisticRegressionModel
from train import train
from test import test

# Load data
df = pd.read_csv('dataset.csv')
df['label'] = df['label'].map({'active': 1, 'passive': 0})

X = df[['query_number', 'confidence', 'label']].values
y = df['is_correct'].values

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegressionModel(input_dim=3, output_dim=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    accuracy = (preds == y_test).float().mean()

print(f"Test Accuracy: {accuracy:.2f}")