import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from logRegressionModel import LogisticRegressionModel
from train import train
from test import test_with_uncertainty

CONFIDENCE_THRESHOLD = 0.75  # Ask a question if confidence is below this
EPOCHS = 50
BATCH_SIZE = 16
LR = 0.01

# ── Load & prep data ──────────────────────────────────────────────────────────
# Parse simulation JSON into (query_number, confidence, label, is_correct) rows.
# learning_type 'active'  → robot self-initiates queries (SNOWMAN, ALIEN)
# learning_type 'passive' → robot only responds, no self-initiated queries (HOUSE, ICE_CREAM)
with open('simulation_data.json') as f:
    sim = json.load(f)

rows = []
for session in sim['sessions']:
    learning_type = session['learning_type']   # 'active' or 'passive'
    for step in session['steps']:
        rows.append({
            'query_number': step['interaction_step'],
            'confidence':   step['robot_confidence'],
            'label':        learning_type,
            'is_correct':   step['robot_correct'],
        })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} interaction steps from simulation_data.json")
print(df.groupby(['label', 'is_correct']).size().to_string(), "\n")

df['label'] = df['label'].map({'active': 1, 'passive': 0})

X = torch.tensor(df[['query_number', 'confidence', 'label']].values, dtype=torch.float32)
y = torch.tensor(df['is_correct'].values, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=1,          shuffle=False)

# ── Model setup ───────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = LogisticRegressionModel(input_dim=3, output_dim=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# ── Training loop (concept-by-concept) ───────────────────────────────────────
print("=" * 50)
print("TRAINING PHASE")
print("=" * 50)
for epoch in range(1, EPOCHS + 1):
    avg_loss, accuracy = train(model, train_loader, criterion, optimizer, device)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.4f}")

# ── Testing phase with uncertainty + active questioning ───────────────────────
print("\n" + "=" * 50)
print("TESTING PHASE  (confidence threshold = {:.0%})".format(CONFIDENCE_THRESHOLD))
print("=" * 50)
test_with_uncertainty(model, test_loader, criterion, device, CONFIDENCE_THRESHOLD)