import torch
def train(model, train_loader, criterion, optimizer, device):
    """
    One full epoch over train_loader.
    Returns (avg_loss, accuracy) so the caller can log progress.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += output.argmax(dim=1).eq(target).sum().item()
        total      += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy