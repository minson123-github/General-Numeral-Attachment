import math
import torch

class sentenceAttention(torch.nn.Module):
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(sentenceAttention, self).__init__()
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.embed_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.atten = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, sent_embed):
        query = self.q_proj(sent_embed)
        sent_embed = self.embed_proj(sent_embed)
        query = torch.mean(query, 1, True)
        attn_output, attn_output_weights = self.atten(query, sent_embed, sent_embed)
        return torch.squeeze(attn_output, dim=1)

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = torch.nn.parameter.Parameter(pe)
        #self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerLayer(torch.nn.Module):

    def __init__(self, init_dim, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj = torch.nn.Sequential(
            torch.nn.Linear(init_dim, embed_dim * 4), 
            torch.nn.ReLU(), 
            torch.nn.Linear(embed_dim * 4, embed_dim)
        )
        self.pos_encode = PositionalEncoding(embed_dim, dropout)
        self.atten_layer = sentenceAttention(embed_dim, num_heads, dropout)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4), 
            torch.nn.ReLU(), 
            torch.nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, sent_embed):
        sent_embed = self.embed_proj(sent_embed)
        sent_embed = sent_embed * math.sqrt(self.embed_dim)
        sent_embed = self.pos_encode(sent_embed)
        outputs = self.atten_layer(sent_embed)
        out1 = self.layernorm1(outputs)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout(ffn_out)
        return self.layernorm2(out1 + ffn_out)

class seq2seqLayer(torch.nn.Module):
    
    def __init__(self, embed_dim: int, mid_dim: int, dropout=0.7):
        super().__init__()
        self.lstm = torch.nn.LSTM(embed_dim, embed_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.linear = torch.nn.Sequential(
            #torch.nn.ReLU(), 
            torch.nn.Linear(embed_dim, 1), 
        )

    def forward(self, inputs, hidden):
        if type(hidden) == type(None):
            outputs, hidden = self.lstm(torch.unsqueeze(inputs, dim=1))
        else:
            outputs, hidden = self.lstm(torch.unsqueeze(inputs, dim=1), hidden)
        outputs = self.linear(outputs)
        return outputs, hidden
