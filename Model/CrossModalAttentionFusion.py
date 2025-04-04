import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):

    def __init__(self, model_dim, num_heads=8):
        super(CrossModalAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        head_dim = model_dim // num_heads

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim,
                                                    num_heads=num_heads,
                                                    batch_first=True)

        self.output_linear = nn.Linear(model_dim, model_dim)

    def forward(self, TT, TA, TV):
        features = torch.stack((TT, TA, TV),
                               dim=1)  # (batch_size, 3, seq_len, model_dim)
        batch_size, num_modalities, seq_len, model_dim = features.size()

        features = features.view(batch_size * num_modalities, seq_len,
                                 model_dim)

        Q = self.q_linear(features)
        K = self.k_linear(features)
        V = self.v_linear(features)

        attn_output, _ = self.multihead_attn(query=Q, key=K, value=V)

        attn_output = attn_output.view(batch_size, num_modalities, seq_len,
                                       model_dim)

        fused_features = torch.mean(attn_output, dim=1)

        output = self.output_linear(fused_features)

        return output
