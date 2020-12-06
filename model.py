import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        # self.hidden_dim = 1024
        self.hidden_dim = 256
        # self.num_layers = 3
        self.num_layers = 2

        # 字典长度
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0,
            # dropout=0.2,
        )
        self.fc = nn.Linear(self.hidden_dim, n_vocab)

        # self.fc1 = nn.Linear(self.hidden_dim, 2048)
        # self.fc2 = nn.Linear(2048, 4096)
        # self.fc3 = nn.Linear(4096, n_vocab)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embed = self.embedding(input)
        output, hidden = self.lstm(embed, (h_0, c_0))

        # output = self.fc(output)
        output = self.fc(output.view(batch_size * seq_len, -1))
        # output = torch.tanh(self.fc1(output))
        # output = torch.tanh(self.fc2(output))
        # output = self.fc3(output)
        # output = output.reshape(batch_size * seq_len, -1)

        return output, hidden

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
