from Model.DialogueRNN import BiModel
import torch
import torch.nn as nn


class SingleModality(nn.Module):

    def __init__(self, dataset, text_dim, audio_dim, visual_dim, model_dim,
                 dropout, n_classes, n_speakers, device):
        super().__init__()

        D_g = D_p = D_e = D_h = D_a = model_dim
        context_attention = 'simple'
        dropout_rec = 0
        listener_state = False

        self.dataset = dataset

        # self.text_fc = nn.Sequential(nn.Linear(text_dim, model_dim * 2),
        #                              nn.Linear(model_dim * 2, model_dim))
        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                                        n_classes, n_speakers, listener_state,
                                        context_attention, D_a, dropout_rec,
                                        dropout, device)

        # self.audio_fc = nn.Sequential(nn.Linear(audio_dim, model_dim * 2),
        #                               nn.Linear(model_dim * 2, model_dim))
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h,
                                         dataset, n_classes, n_speakers,
                                         listener_state, context_attention,
                                         D_a, dropout_rec, dropout, device)

        # self.visual_fc = nn.Sequential(nn.Linear(visual_dim, model_dim * 2),
        #                                nn.Linear(model_dim * 2, model_dim))
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h,
                                          dataset, n_classes, n_speakers,
                                          listener_state, context_attention,
                                          D_a, dropout_rec, dropout, device)

        self.modality_weight_layer = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(model_dim, 3))

    def forward(self, text_features, audio_features, visual_features,
                speaker_masks, utterance_masks):
        # text_features = self.text_fc(texts)
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(text_features, speaker_masks,
                                                  utterance_masks)
        # text_features = self.text_dialoguernn(text_features, speaker_masks,
        #                                           utterance_masks)

        # audio_features = self.audio_fc(audios)
        audio_features = self.audio_dialoguernn(audio_features, speaker_masks,
                                                utterance_masks)

        # visual_features = self.visual_fc(visuals)
        visual_features = self.visual_dialoguernn(visual_features,
                                                  speaker_masks,
                                                  utterance_masks)

        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        return text_features, audio_features, visual_features
