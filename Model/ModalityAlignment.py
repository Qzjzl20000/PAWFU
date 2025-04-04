import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.CMDLoss import CMDLoss
from Utils.AdaIN import AdaIN


class ModalityAlignment(nn.Module):

    def __init__(self, model_dim, hidden_dim):
        super(ModalityAlignment, self).__init__()
        self.if_use_adaIN = False

        self.T_mlp = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(hidden_dim, model_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(model_dim, model_dim))
        self.A_mlp = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(hidden_dim, model_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(model_dim, model_dim))
        self.V_mlp = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(hidden_dim, model_dim), nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(model_dim, model_dim))

        if self.if_use_adaIN:
            self.adain = AdaIN()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_features, audio_features, visual_features):

        if self.if_use_adaIN:
            text_features_aligned = self.adain(self.T_mlp(text_features))
            audio_features_aligned = self.adain(self.A_mlp(audio_features))
            visual_features_aligned = self.adain(self.V_mlp(visual_features))
        else:
            text_features_aligned = self.T_mlp(text_features)
            audio_features_aligned = self.A_mlp(audio_features)
            visual_features_aligned = self.V_mlp(visual_features)

        return text_features_aligned, audio_features_aligned, visual_features_aligned

    def reconstruct(self, text_features, audio_features, visual_features):
        # texts_recon = self.T_recon_mlp(text_features)
        # audios_recon = self.A_recon_mlp(audio_features)
        # visuals_recon = self.V_recon_mlp(visual_features)
        # return texts_recon, audios_recon, visuals_recon

        texts_to_texts_recon = self.T_recon_T_mlp(text_features)
        texts_to_audios_recon = self.T_recon_A_mlp(text_features)
        texts_to_visuals_recon = self.T_recon_V_mlp(text_features)

        audios_to_audios_recon = self.A_recon_A_mlp(text_features)
        audios_to_texts_recon = self.A_recon_T_mlp(audio_features)
        audios_to_visuals_recon = self.A_recon_V_mlp(audio_features)

        visuals_to_visuals_recon = self.V_recon_V_mlp(text_features)
        visuals_to_texts_recon = self.V_recon_T_mlp(visual_features)
        visuals_to_audios_recon = self.V_recon_A_mlp(visual_features)

        return texts_to_texts_recon, texts_to_audios_recon, texts_to_visuals_recon, audios_to_audios_recon, audios_to_texts_recon, audios_to_visuals_recon, visuals_to_visuals_recon, visuals_to_texts_recon, visuals_to_audios_recon

    def recon_loss(self, texts_pre_align, audios_pre_align, visuals_pre_align,
                   text_features, audio_features, visual_features,
                   GroundTruth):

        # Single-Modal Cycle Consistency Loss Control feature-self MSE
        texts_recon_loss = self.MSE_loss(text_features, texts_pre_align)
        audios_recon_loss = self.MSE_loss(audio_features, audios_pre_align)
        visuals_recon_loss = self.MSE_loss(visual_features, visuals_pre_align)
        single_CC_recon_loss = texts_recon_loss + audios_recon_loss + visuals_recon_loss
        # Multi-Modal Cross Semantic Loss Control Cross Aligned Feature
        cmd_loss = CMDLoss(kernel='rbf', gamma=1.0)
        cs_loss_TA = cmd_loss(text_features, audio_features, GroundTruth)
        cs_loss_TV = cmd_loss(text_features, visual_features, GroundTruth)
        cs_loss_AV = cmd_loss(audio_features, visual_features, GroundTruth)
        mm_CS_recon_loss = cs_loss_TA + cs_loss_TV + cs_loss_AV
        return single_CC_recon_loss, mm_CS_recon_loss

    def cosine_similarity_loss(self, original_features, processed_features):
        feature_dim = original_features.size(-1)
        original_features_flat = original_features.reshape(-1, feature_dim)
        processed_features_flat = processed_features.reshape(-1, feature_dim)
        similarity_score = F.cosine_similarity(original_features_flat,
                                               processed_features_flat,
                                               dim=-1)
        similarity_score = similarity_score.mean(dim=-1)
        # print("similarity_score", similarity_score)
        semantic_loss = 1 - similarity_score
        return semantic_loss

    def MSE_loss(self, original_features, processed_features):
        feature_dim = original_features.size(-1)
        original_features_flat = original_features.reshape(-1, feature_dim)
        processed_features_flat = processed_features.reshape(-1, feature_dim)
        mse_loss = F.mse_loss(original_features_flat, processed_features_flat)
        return mse_loss
