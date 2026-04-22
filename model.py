import torch.nn as nn

class Architecture:
    input_dim : int
    hidden_dims : list[int]
    output_dim : int
    dropout : list[float]

    def __init__(self, input_dim : int, hidden_dims : list[int], output_dim : int, dropout : list[float] = None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout if dropout is not None else [0.0] * len(hidden_dims)

    def validate(self):
        assert self.input_dim > 0, "Input dimension must be greater than 0"
        assert self.output_dim > 0, "Output dimension must be greater than 0"

        assert len(self.hidden_dims) == len(self.dropout), "Length of hidden_dims and dropout must be the same"
        for dim in self.hidden_dims:
            assert dim > 0, "Hidden dimensions must be greater than 0"
        
        for dim in self.dropout:
            assert 0.0 <= dim < 1.0, "Dropout values must be in the range [0.0, 1.0)"
        

class SimpleMLP(nn.Module):
    def __init__(self, arch: Architecture):
        super().__init__()

        arch.validate()
        layers = []
        layers.append(nn.Flatten())
        for i in range(len(arch.hidden_dims)):
            layers.append(nn.Linear(arch.hidden_dims[i-1] if i > 0 else arch.input_dim, arch.hidden_dims[i]))
            layers.append(nn.ReLU())
            if i < len(arch.hidden_dims) - 1:
                layers.append(nn.Dropout(arch.dropout[i]))
        layers.append(nn.Linear(arch.hidden_dims[-1], arch.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, arch: Architecture):
        super().__init__()

        arch.validate()
        layers = []
        pooling_count = 0
        for i in range(len(arch.hidden_dims)):
            layers.append(nn.Conv2d(arch.hidden_dims[i-1] if i > 0 else arch.input_dim, arch.hidden_dims[i], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if pooling_count < 3:
                layers.append(nn.MaxPool2d(kernel_size=2))
                pooling_count += 1
            if i < len(arch.hidden_dims) - 1:
                layers.append(nn.Dropout2d(arch.dropout[i]))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(arch.hidden_dims[-1] * (28 // (2 ** pooling_count)) ** 2, arch.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
