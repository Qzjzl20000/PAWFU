import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)

    def scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim=-1), V)

        return attention

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        cross_attention = self.scaled_dot_product_attention(Q, K, V)

        return cross_attention


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            CrossAttention(model_dim, Q_dim, K_dim, V_dim)
            for _ in range(self.num_heads)
        ])
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)

    def forward(self, query, key, value):
        heads = [
            self.attention_heads[i](query, key, value)
            for i in range(self.num_heads)
        ]
        multihead_cross_attention = self.projection_matrix(
            torch.cat(heads, dim=-1))

        return multihead_cross_attention


class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))


class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))
        return output


class CrossModalFusion(nn.Module):

    def __init__(self, model_dim, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.cross_attention = MultiHeadCrossAttention(num_heads, model_dim,
                                                       Q_dim, K_dim, V_dim)
        self.add_norm = AddNorm(model_dim, dropout_rate)
        self.ffn = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_ffn = AddNorm(model_dim, dropout_rate)

    def forward(self, query, key, value):
        cross_att = self.cross_attention(query, key, value)
        x = self.add_norm(query, cross_att)
        ffn_output = self.ffn(x)
        output = self.add_norm_ffn(x, ffn_output)
        return output
