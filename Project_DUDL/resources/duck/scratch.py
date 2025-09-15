class NN(nn.Module):
    self.__init__(features,
                  hidden_layers=num_layers,
                  neurons=num_neurons,
                  features=df.shape[1],
                  predictions)
    super().__init__()
    self.dict_layers = nn.ModuleDict()
    self.hidden_layers = hidden_layers

    self.dict_layers['input'] = nn.Linear(features, neurons)

    for _ in range(self.hidden_layers):
        self.dict_layers[f'hidden{_}'] = nn.Linear(neurons, neurons)

    self.dict_layers['output'] = nn.Linear(neurons, predictions)

    def forward(self, x):
        
        x = F.relu(self.dict_layers['input'](x))

        for _ in range(self.hidden_layers):
            x = F.relu(self.dict_layers[f'hidden{_}'])(x)

        x = self.dict_layers['output'](x)