from torch import nn
class SimpleCNN(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2)
        ])
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(nn.Flatten(),
                                nn.Linear(2304, out_features))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
        x = self.head(x)
        return x