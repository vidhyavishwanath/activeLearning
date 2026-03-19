def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')