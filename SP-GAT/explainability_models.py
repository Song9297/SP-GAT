"""Import-safe definitions of the four project models.

Tests compare class ASTs with the original scripts to detect architecture drift.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
GAT_LAYERS = 2
DROPOUT = 0.2
EDGE_ATTR_DIM = 4
n_bands = 343


# Source: resnet1d_agb.py

class ResBlock1D(nn.Module):
    """Basic 1D residual block with optional downsample shortcut."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 dropout=0.0, use_bn=True):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)
        self.dp    = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dp(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet1D(nn.Module):


    def __init__(self, in_ch=1, block_chs=[32, 32, 64, 64],
                 kernel_size=7, dropout=0.2, use_bn=True):
        super().__init__()
        self.dropout = dropout

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, block_chs[0], kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(block_chs[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
        )

        blocks = []
        for i in range(len(block_chs) - 1):
            ic, oc = block_chs[i], block_chs[i + 1]
            st = 1 if ic == oc else 2
            blocks.append(ResBlock1D(ic, oc, kernel_size=3, stride=st,
                                     dropout=dropout, use_bn=use_bn))
        self.res_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        fc_in = block_chs[-1]
        self.regressor = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.regressor(x).squeeze(-1)

# Source: transformer_agb.py

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder block."""
    def __init__(self, d_model, nheads, ff_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):

        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        x = x + self.dp(attn_out)

        x = x + self.dp(self.ff(self.norm2(x)))
        if return_attention:
            return x, attn_weights
        return x

class SpectralTransformer(nn.Module):


    def __init__(self, seq_len=n_bands, d_model=64, nheads=4,
                 nlayers=2, ff_dim=128, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model


        self.embed = nn.Linear(1, d_model)
        self.pe = PositionalEncoding(d_model, max_len=seq_len + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model, nheads, ff_dim, dropout)
            for _ in range(nlayers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_attention=False):

        B = x.size(0)
        tokens = self.embed(x.unsqueeze(-1))      # (B, 343, d_model)

        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 344, d_model)
        tokens = self.pe(tokens)

        attn_list = []
        if return_attention:
            for block in self.encoder:
                tokens, aw = block(tokens, return_attention=True)
                attn_list.append(aw)
        else:
            tokens = self.encoder(tokens)

        tokens = self.final_norm(tokens)

        cls_out  = tokens[:, 0, :]                # (B, d_model)
        mean_out = tokens[:, 1:, :].mean(dim=1)   # (B, d_model)
        pooled = torch.cat([cls_out, mean_out], dim=-1)  # (B, 2*d_model)

        out = self.regressor(pooled).squeeze(-1)
        if return_attention:
            return out, attn_list
        return out

# Source: gat_agb.py

class GATBaseline(nn.Module):

    def __init__(self, in_channels=2, hidden_dim=HIDDEN_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=GAT_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GATConv(in_channels, hidden_dim // num_heads, num_heads,
                                       concat=True, dropout=dropout,
                                       add_self_loops=False))

        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                           concat=True, dropout=dropout,
                                           add_self_loops=False))

        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, 1, concat=False, dropout=dropout,
                                       add_self_loops=False))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        attn_w_list = []
        for gat_layer in self.gat_layers:
            if return_attention:
                x, (attn_ei, attn_w) = gat_layer(x, edge_index,
                                                  return_attention_weights=True)
                attn_w_list.append((attn_ei, attn_w))
            else:
                x = gat_layer(x, edge_index)
            x = nn.functional.elu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.regressor(x)
        if return_attention:
            return x.squeeze(-1), attn_w_list
        return x.squeeze(-1)

# Source: PP-SGAT.py

class GATRegressorVI(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=HIDDEN_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=GAT_LAYERS,
                 dropout=DROPOUT, edge_attr_dim=EDGE_ATTR_DIM):
        super().__init__()
        self.dropout = dropout
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels, hidden_dim // num_heads, num_heads,
                                       concat=True, dropout=dropout, edge_dim=edge_attr_dim,
                                       add_self_loops=True))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                           concat=True, dropout=dropout, edge_dim=edge_attr_dim,
                                           add_self_loops=True))
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, 1, concat=False, dropout=dropout,
                                       edge_dim=edge_attr_dim, add_self_loops=True))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        attn_w_list = []
        for gat_layer in self.gat_layers:
            if return_attention:
                x, (attn_ei, attn_w) = gat_layer(x, edge_index, edge_attr=edge_attr,
                                                  return_attention_weights=True)
                attn_w_list.append((attn_ei, attn_w))
            else:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = nn.functional.elu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.regressor(x)
        if return_attention:
            return x.squeeze(-1), attn_w_list
        return x.squeeze(-1)
