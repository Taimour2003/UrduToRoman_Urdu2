
import torch
import torch.nn as nn

PAD = 0

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat   = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        hidden_dec = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0).repeat(4,1,1)
        cell_dec   = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0).repeat(4,1,1)
        return outputs, hidden_dec, cell_dec

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        embedded = self.dropout(self.embedding(trg))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        preds = self.fc_out(outputs)
        return preds, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        _, hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg[:, :-1], hidden, cell)
        return outputs
