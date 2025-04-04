import torch
import torch.nn as nn


class PreAlignment(nn.Module):

    def __init__(self,
                 text_dim,
                 audio_dim,
                 visual_dim,
                 model_dim,
                 if_use_LN=False,
                 if_use_conv1d=False):
        super(PreAlignment, self).__init__()
        self.if_use_LN = if_use_LN
        self.if_use_conv1d = if_use_conv1d

        if if_use_LN:
            self.text_layer_norm = nn.LayerNorm(text_dim)
            self.audio_layer_norm = nn.LayerNorm(audio_dim)
            self.visual_layer_norm = nn.LayerNorm(visual_dim)

        # pre_align into same dimension
        if if_use_conv1d:
            self.text_conv_1d = nn.Conv1d(text_dim,
                                          model_dim,
                                          kernel_size=1,
                                          padding=0,
                                          bias=False)
            self.audio_conv_1d = nn.Conv1d(audio_dim,
                                           model_dim,
                                           kernel_size=1,
                                           padding=0,
                                           bias=False)
            self.visual_conv_1d = nn.Conv1d(visual_dim,
                                            model_dim,
                                            kernel_size=1,
                                            padding=0,
                                            bias=False)
        else:
            self.text_pre_align_MLP = nn.Sequential(
                nn.Linear(text_dim, model_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(model_dim, model_dim))
            self.audio_pre_align_MLP = nn.Sequential(
                nn.Linear(audio_dim, model_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(model_dim, model_dim))
            self.visual_pre_align_MLP = nn.Sequential(
                nn.Linear(visual_dim, model_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(model_dim, model_dim))

    def forward(self, texts, audios, visuals):
        if self.if_use_LN:
            texts = self.text_layer_norm(texts)
            audios = self.audio_layer_norm(audios)
            visuals = self.visual_layer_norm(visuals)

        if self.if_use_conv1d:
            text_features = self.text_conv_1d(texts.permute(0, 2,
                                                            1)).transpose(
                                                                1, 2)
            audio_features = self.audio_conv_1d(audios.permute(0, 2,
                                                               1)).transpose(
                                                                   1, 2)
            visual_features = self.visual_conv_1d(visuals.permute(
                0, 2, 1)).transpose(1, 2)
        else:
            text_features = self.text_pre_align_MLP(texts)
            audio_features = self.audio_pre_align_MLP(audios)
            visual_features = self.visual_pre_align_MLP(visuals)

        return text_features, audio_features, visual_features
