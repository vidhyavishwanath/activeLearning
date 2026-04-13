import torch

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        """Returns predicted class indices without gradients."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def confidence(self, x):
        """Returns softmax probabilities — useful for uncertainty scoring in test.py."""
        with torch.no_grad():
            return torch.nn.functional.softmax(self.forward(x), dim=1)