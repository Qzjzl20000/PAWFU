import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGatedFusion(nn.Module):

    def __init__(self, feature_dim, num_heads=8, attn_dim=512, output_dim=256):
        """
        动态自适应门控机制，添加线性层对输入特征进行对齐
        :param feature_dim: 原始输入特征维度
        :param num_heads: 多头注意力头数
        :param attn_dim: 自注意力的嵌入维度
        :param output_dim: 输出特征维度
        """
        super(AdaptiveGatedFusion, self).__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.output_dim = output_dim

        # 线性层将输入特征对齐到 attention 所需的维度
        self.feature_proj = nn.Linear(feature_dim, attn_dim)

        # 自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=attn_dim,
                                               num_heads=num_heads)

        # 门控权重生成器
        self.gate_gen_attention = nn.MultiheadAttention(embed_dim=attn_dim * 2,
                                                        num_heads=num_heads)

        # 将门控权重维度转换为 attn_dim
        self.gate_proj = nn.Linear(attn_dim * 2, attn_dim)

        # 门控特征映射
        self.feature_transform = nn.Linear(attn_dim, output_dim)

    def forward(self, features):
        """
        :param features: 输入模态特征 (batch_size, seq_len, feature_dim)
        :return: 门控后的模态特征 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = features.size()

        # 通过线性层对齐到 attention 所需的维度
        features_aligned = self.feature_proj(
            features)  # (batch_size, seq_len, attn_dim)

        # 自注意力生成上下文信息
        features_flat = features_aligned.transpose(
            0, 1)  # (seq_len, batch_size, attn_dim)
        attn_output, _ = self.attention(
            features_flat, features_flat,
            features_flat)  # (seq_len, batch_size, attn_dim)
        context_emb = torch.mean(attn_output,
                                 dim=0)  # 聚合上下文信息 (batch_size, attn_dim)

        # 拼接特征和上下文信息
        context_expanded = context_emb.unsqueeze(1).expand(
            -1, seq_len, -1)  # (batch_size, seq_len, attn_dim)

        combined_input = torch.cat(
            [features_aligned, context_expanded],
            dim=-1)  # (batch_size, seq_len, attn_dim * 2)

        # 多头注意力生成门控权重
        combined_input_flat = combined_input.transpose(
            0, 1)  # (seq_len, batch_size, attn_dim * 2)
        gate_weights, _ = self.gate_gen_attention(
            combined_input_flat, combined_input_flat,
            combined_input_flat)  # (seq_len, batch_size, attn_dim * 2)
        gate_weights = gate_weights.transpose(
            0, 1)  # (batch_size, seq_len, attn_dim * 2)

        # 将门控权重维度转换为 attn_dim
        gate_weights = self.gate_proj(
            gate_weights)  # (batch_size, seq_len, attn_dim)

        # 应用门控权重到特征
        gated_features = gate_weights * features_aligned  # (batch_size, seq_len, attn_dim)

        # 映射回输出特征维度
        output_features = F.relu(self.feature_transform(
            gated_features))  # (batch_size, seq_len, output_dim)
        return output_features


class Unimodal_GatedFusion(nn.Module):

    def __init__(self, hidden_size):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep


class Multimodal_GatedFusion(nn.Module):

    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([
            self.fc(a).unsqueeze(-2),
            self.fc(b).unsqueeze(-2),
            self.fc(c).unsqueeze(-2)
        ],
                              dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep
