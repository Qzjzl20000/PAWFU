import torch.nn as nn
from Model.ModalityAlignment import ModalityAlignment
from Model.PreAlignment import PreAlignment


class AlignModule(nn.Module):

    def __init__(self, text_dim, audio_dim, visual_dim, model_dim, hidden_dim):
        super(AlignModule, self).__init__()

        # Pre Align
        self.pre_align = PreAlignment(text_dim,
                                      audio_dim,
                                      visual_dim,
                                      model_dim,
                                      if_use_LN=False,
                                      if_use_conv1d=False)

        # Modality Features Align
        self.modality_align_gen = ModalityAlignment(model_dim, hidden_dim)

    def feature_alignment_generate(self, texts, audios, visuals):
        text_features, audio_features, visual_features = self.modality_align_gen(
            texts, audios, visuals)
        return text_features, audio_features, visual_features
