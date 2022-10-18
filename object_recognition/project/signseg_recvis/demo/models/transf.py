import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy

def format_predictions_output(predictions: torch.Tensor) -> torch.Tensor:
    ret =  predictions.permute(1,2,0)
    return ret 

class TransformerModel(nn.Module):
    # def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,nlayers: int, dropout: float = 0.5):
    def __init__(self, d_model, nhead, nlayers,dim_feedforward_encoder, dropout):
        ntoken = 2
        # d_model = 1024
        # nhead = 8
        # nlayers = 8
        # dropout = 0.4

        self.d_model = d_model
        
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward_encoder, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
       
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.permute(2,0,1)
        src = src * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        output = self.decoder(output)
        return format_predictions_output(output)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
