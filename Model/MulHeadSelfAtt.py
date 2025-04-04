import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, model_dim, attention_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, attention_dim)
        self.key_matrix = nn.Linear(model_dim, attention_dim)
        self.value_matrix = nn.Linear(model_dim, attention_dim)

    def scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim=-1), V)

        return attention

    def forward(self, x):
        Q = self.query_matrix(x)
        K = self.key_matrix(x)
        V = self.value_matrix(x)
        self_attention = self.scaled_dot_product_attention(Q, K, V)

        return self_attention


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, num_heads, model_dim, attention_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            SelfAttention(model_dim, attention_dim)
            for _ in range(self.num_heads)
        ])
        self.projection_matrix = nn.Linear(num_heads * attention_dim,
                                           model_dim)

    def forward(self, x):
        heads = [self.attention_heads[i](x) for i in range(self.num_heads)]
        multihead_self_attention = self.projection_matrix(
            torch.cat(heads, dim=-1))

        return multihead_self_attention


class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_x):
        return self.norm(x + self.dropout(sublayer_x))


class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))


class SingleModelSelfAttn(nn.Module):

    def __init__(
        self,
        model_dim,
        target_dim,
        layer_num,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super(SingleModelSelfAttn, self).__init__()
        self.layer_num = layer_num
        attention_dim = model_dim // num_heads
        self.self_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(num_heads, model_dim, attention_dim)
            for _ in range(layer_num)
        ])
        self.add_norm_layers = nn.ModuleList(
            [AddNorm(model_dim, dropout_rate) for _ in range(layer_num)])
        self.ffn_layers = nn.ModuleList([
            Feedforward(model_dim, model_dim, dropout_rate)
            for _ in range(layer_num)
        ])
        self.add_norm_ffn_layers = nn.ModuleList(
            [AddNorm(model_dim, dropout_rate) for _ in range(layer_num)])
        self.project_to_target_dim = nn.Linear(model_dim, target_dim)

    def forward(self, x):
        for i in range(self.layer_num):
            self_att = self.self_attention_layers[i](x)
            x = self.add_norm_layers[i](x, self_att)
            x = self.add_norm_ffn_layers[i](x, self.ffn_layers[i](x))
        output_projected = self.project_to_target_dim(x)
        return output_projected


class SingleModelSelfAttnModel(nn.Module):

    def __init__(self,
                 text_dim=768,
                 audio_dim=512,
                 visual_dim=1000,
                 target_dim=512,
                 num_heads=8,
                 dropout_rate=0.1,
                 layer_num=3):
        super(SingleModelSelfAttnModel, self).__init__()

        self.text_self_attn = SingleModelSelfAttn(text_dim, target_dim,
                                                  layer_num, num_heads,
                                                  dropout_rate)
        self.audio_self_attn = SingleModelSelfAttn(audio_dim, target_dim,
                                                   layer_num, num_heads,
                                                   dropout_rate)
        self.visual_self_attn = SingleModelSelfAttn(visual_dim, target_dim,
                                                    layer_num, num_heads,
                                                    dropout_rate)

    def forward(self, text, audio, visual):
        text_output = self.text_self_attn(text)
        audio_output = self.audio_self_attn(audio)
        visual_output = self.visual_self_attn(visual)

        return text_output, audio_output, visual_output
