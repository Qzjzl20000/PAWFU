import torch
import torch.nn as nn


class DualModality(nn.Module):

    def __init__(self,
                 model_dim,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 dropout_rate=0.1):
        super(DualModality, self).__init__()

        self.use_layerNorm_first = False
        self.use_SA_first = False
        # self.single_modality_num_layers = 2
        self.single_modality_num_layers = num_layers

        if self.use_layerNorm_first:
            self.layerNorm_T = nn.LayerNorm(model_dim)
            self.layerNorm_A = nn.LayerNorm(model_dim)
            self.layerNorm_V = nn.LayerNorm(model_dim)

        self.encoder_layers = nn.ModuleDict({
            "TT":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate)
                for _ in range(self.single_modality_num_layers)
            ]),
            "AA":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate)
                for _ in range(self.single_modality_num_layers)
            ]),
            "VV":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate)
                for _ in range(self.single_modality_num_layers)
            ]),
            "TA":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ]),
            "TV":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ]),
            "AT":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ]),
            "AV":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ]),
            "VT":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ]),
            "VA":
            nn.ModuleList([
                DualModalityLayer(model_dim, num_heads, hidden_dim,
                                  dropout_rate) for _ in range(num_layers)
            ])
        })

    def forward(self, text_features, audio_features, visual_features):
        if self.use_layerNorm_first:
            text_features = self.layerNorm_T(text_features)
            audio_features = self.layerNorm_A(audio_features)
            visual_features = self.layerNorm_V(visual_features)

        # Single modality interaction (SA)
        tt_output, aa_output, vv_output = text_features.clone(
        ), audio_features.clone(), visual_features.clone()
        for layer in self.encoder_layers["TT"]:
            tt_output = layer(tt_output, text_features, text_features)
        for layer in self.encoder_layers["AA"]:
            aa_output = layer(aa_output, audio_features, audio_features)
        for layer in self.encoder_layers["VV"]:
            vv_output = layer(vv_output, visual_features, visual_features)

        if self.use_SA_first:
            text_features, audio_features, visual_features = tt_output, aa_output, vv_output

        # Cross modality interaction (CA)
        # TA interaction
        ta_output, at_output = text_features.clone(), audio_features.clone()
        for layer in self.encoder_layers["TA"]:
            ta_output = layer(ta_output, audio_features, audio_features)
        for layer in self.encoder_layers["AT"]:
            at_output = layer(at_output, text_features, text_features)

        # TV interaction
        tv_output, vt_output = text_features.clone(), visual_features.clone()
        for layer in self.encoder_layers["TV"]:
            tv_output = layer(tv_output, visual_features, visual_features)
        for layer in self.encoder_layers["VT"]:
            vt_output = layer(vt_output, text_features, text_features)

        # AV interaction
        av_output, va_output = audio_features.clone(), visual_features.clone()
        for layer in self.encoder_layers["AV"]:
            av_output = layer(av_output, visual_features, visual_features)
        for layer in self.encoder_layers["VA"]:
            va_output = layer(va_output, audio_features, audio_features)

        return tt_output, aa_output, vv_output, ta_output, at_output, tv_output, vt_output, av_output, va_output


class DualModalityEnhanced(nn.Module):

    def __init__(self,
                 model_dim,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 dropout_rate=0.1):
        super(DualModalityEnhanced, self).__init__()

        self.encoder_layers = nn.ModuleDict({
            "TA":
            nn.ModuleList([
                MultiHeadCrossAttentionLayer(model_dim, num_heads, hidden_dim,
                                             dropout_rate)
                for _ in range(num_layers)
            ]),
            "TV":
            nn.ModuleList([
                MultiHeadCrossAttentionLayer(model_dim, num_heads, hidden_dim,
                                             dropout_rate)
                for _ in range(num_layers)
            ]),
            "AV":
            nn.ModuleList([
                MultiHeadCrossAttentionLayer(model_dim, num_heads, hidden_dim,
                                             dropout_rate)
                for _ in range(num_layers)
            ])
        })

    def forward(self, text_features, audio_features, visual_features):

        # TA interaction
        new_ta_t, new_ta_a = text_features, audio_features
        for layer in self.encoder_layers["TA"]:
            new_ta_t, new_ta_a = layer(new_ta_t, new_ta_a)
        # TV interaction
        new_tv_t, new_tv_v = text_features, visual_features
        for layer in self.encoder_layers["TV"]:
            new_tv_t, new_tv_v = layer(new_tv_t, new_tv_v)
        # AV interaction
        new_av_a, new_av_v = audio_features, visual_features
        for layer in self.encoder_layers["AV"]:
            new_av_a, new_av_v = layer(new_av_a, new_av_v)

        return new_ta_t, new_ta_a, new_tv_t, new_tv_v, new_av_a, new_av_v


class MultiHeadCrossAttentionLayer(nn.Module):

    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(MultiHeadCrossAttentionLayer, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Q, K, V
        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V1 = nn.Linear(model_dim, model_dim)
        self.W_V2 = nn.Linear(model_dim, model_dim)

        # learnable param alpha and beta
        self.alpha_1 = nn.Parameter(torch.randn(1))
        self.beta_1 = nn.Parameter(torch.randn(1))
        self.alpha_2 = nn.Parameter(torch.randn(1))
        self.beta_2 = nn.Parameter(torch.randn(1))

        # output
        self.W_O1 = nn.Linear(model_dim, model_dim)
        self.W_O2 = nn.Linear(model_dim, model_dim)

        # ffn
        self.ffn1 = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, model_dim))
        self.ffn2 = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, model_dim))

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, modal1, modal2):
        batch_size, seq_len, model_dim = modal1.size()
        Q = self.W_Q(modal1).view(batch_size, seq_len, self.num_heads,
                                  self.head_dim).transpose(1, 2)
        K = self.W_K(modal2).view(batch_size, seq_len, self.num_heads,
                                  self.head_dim).transpose(1, 2)
        V1 = self.W_V1(modal1).view(batch_size, seq_len, self.num_heads,
                                    self.head_dim).transpose(1, 2)
        V2 = self.W_V2(modal2).view(batch_size, seq_len, self.num_heads,
                                    self.head_dim).transpose(1, 2)

        # Q to K
        S1 = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) /
                           (self.head_dim**0.5),
                           dim=-1)
        # K to Q
        S2 = torch.softmax(torch.matmul(K, Q.transpose(-1, -2)) /
                           (self.head_dim**0.5),
                           dim=-1)

        if True:
            new_V1 = S2 @ V1
            new_V2 = S1 @ V2
        else:

            # new_V1 = S2 @ V1 + (1 - S1) @ V1
            # new_V2 = S1 @ V2 + (1 - S2) @ V2
            weights_1 = torch.softmax(torch.stack([self.alpha_1, self.beta_1],
                                                  dim=-1),
                                      dim=-1)
            alpha_1 = weights_1[:, 0].unsqueeze(-1).unsqueeze(-1)
            beta_1 = weights_1[:, 1].unsqueeze(-1).unsqueeze(-1)
            weights_2 = torch.softmax(torch.stack([self.alpha_2, self.beta_2],
                                                  dim=-1),
                                      dim=-1)
            alpha_2 = weights_2[:, 0].unsqueeze(-1).unsqueeze(-1)
            beta_2 = weights_2[:, 1].unsqueeze(-1).unsqueeze(-1)

            # new V
            new_V1 = (alpha_1 * (1 - S1) + beta_1 * S2) @ V1
            new_V2 = (alpha_2 * (1 - S2) + beta_2 * S1) @ V2

        new_V1 = new_V1.transpose(1,
                                  2).contiguous().view(batch_size, seq_len,
                                                       model_dim)
        new_V2 = new_V2.transpose(1,
                                  2).contiguous().view(batch_size, seq_len,
                                                       model_dim)
        output_V1 = self.W_O1(new_V1)
        output_V2 = self.W_O2(new_V2)

        output_V1 = output_V1 + modal1
        output_V2 = output_V2 + modal2

        new_modal1 = self.ffn1(self.layer_norm_1(output_V1)) + output_V1
        new_modal2 = self.ffn2(self.layer_norm_2(output_V2)) + output_V2

        return new_modal1, new_modal2


class DualModalityLayer(nn.Module):

    def __init__(self, d_model, nhead, hidden_dim, dropout=0.1):
        super(DualModalityLayer, self).__init__()
        # self.layer_norm_1 = nn.LayerNorm(d_model)
        # self.layer_norm_2 = nn.LayerNorm(d_model)
        # self.layer_norm_3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = Feedforward(d_model, hidden_dim, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, Q, K, V):

        # Transpose to (seq_len, batch, model_dim) for MultiheadAttention
        Q_transposed = Q.transpose(0, 1)
        K_transposed = K.transpose(0, 1)
        V_transposed = V.transpose(0, 1)
        # Q_transposed = self.layer_norm_1(Q).transpose(0, 1)
        # K_transposed = self.layer_norm_2(K).transpose(0, 1)
        # V_transposed = self.layer_norm_3(V).transpose(0, 1)

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
