import torch.nn as nn


class CrossFusion(nn.Module):

    def __init__(self,
                 model_dim,
                 dim_times,
                 num_heads,
                 hidden_dim,
                 dropout_rate=0.1):
        super(CrossFusion, self).__init__()
        self.cross_fusion = CrossModalityFusionLayer(model_dim * dim_times,
                                                     num_heads, hidden_dim,
                                                     dropout_rate)

    def forward(self, input_features):
        fusion_output = self.cross_fusion(input_features, input_features,
                                          input_features)
        return fusion_output


class CrossModalityFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, hidden_dim, dropout=0.1):
        super(CrossModalityFusionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = Feedforward(d_model, hidden_dim, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, Q, K, V):

        # Transpose to (seq_len, batch, model_dim) for MultiheadAttention
        Q_transposed = Q.transpose(0, 1)
        K_transposed = K.transpose(0, 1)
        V_transposed = V.transpose(0, 1)

        attn_output, _ = self.self_attn(Q_transposed, K_transposed,
                                        V_transposed)

        # Transpose back to (batch, seq_len, model_dim)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.add_norm_1(Q, lambda x: attn_output)
        ff_output = self.feed_forward(attn_output)
        output = self.add_norm_2(attn_output, lambda x: ff_output)
        return output


class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(model_dim, hidden_dim), nn.LeakyReLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, model_dim))

    def forward(self, x):
        return self.feed_forward_layer(x)


class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))
        return output
