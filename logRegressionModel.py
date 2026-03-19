import torch

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            predicted = torch.argmax(outputs, dim=1)
        return predicted
    
    def backward(self, x, y_true):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        optimizer.zero_grad()
        outputs = self.forward(x)
        loss = criterion(outputs, y_true)
        loss.backward()
        optimizer.step()

        return loss.item()
    
