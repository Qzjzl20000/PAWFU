import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdversarialModule(nn.Module):

    def __init__(self, model_dim, hidden_dim):
        super(DomainAdversarialModule, self).__init__()
        self.if_add_modality_ids = False
        if self.if_add_modality_ids:
            model_dim += 1
        # self.domain_discriminator = nn.Sequential(
        #     nn.Linear(model_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, model_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(model_dim, model_dim // 2),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(model_dim // 2, 3)  # TAV
        # )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, 3)  # TAV
        )

    def domain_adversarial_loss(self, text_features_aligned,
                                audio_features_aligned,
                                visual_features_aligned):
        batch_size, seq_len, _ = text_features_aligned.size()
        device = text_features_aligned.device

        if self.if_add_modality_ids:

            modality_ids_text = torch.zeros(
                (batch_size * seq_len, 1)).to(device)
            modality_ids_audio = torch.ones(
                (batch_size * seq_len, 1)).to(device)
            modality_ids_visual = 2 * torch.ones(
                (batch_size * seq_len, 1)).to(device)

            text_domain_input = torch.cat([
                text_features_aligned.reshape(
                    -1, text_features_aligned.size(-1)), modality_ids_text
            ],
                                          dim=-1)
            audio_domain_input = torch.cat([
                audio_features_aligned.reshape(
                    -1, audio_features_aligned.size(-1)), modality_ids_audio
            ],
                                           dim=-1)
            visual_domain_input = torch.cat([
                visual_features_aligned.reshape(
                    -1, visual_features_aligned.size(-1)), modality_ids_visual
            ],
                                            dim=-1)
        else:
            text_domain_input = text_features_aligned.reshape(
                -1, text_features_aligned.size(-1))
            audio_domain_input = audio_features_aligned.reshape(
                -1, audio_features_aligned.size(-1))
            visual_domain_input = visual_features_aligned.reshape(
                -1, visual_features_aligned.size(-1))

        domain_loss_text = F.cross_entropy(
            self.domain_discriminator(text_domain_input),
            torch.zeros(batch_size * seq_len).long().to(device))
        domain_loss_audio = F.cross_entropy(
            self.domain_discriminator(audio_domain_input),
            torch.ones(batch_size * seq_len).long().to(device))
        domain_loss_visual = F.cross_entropy(
            self.domain_discriminator(visual_domain_input),
            2 * torch.ones(batch_size * seq_len).long().to(device))

        adv_loss = (domain_loss_text + domain_loss_audio +
                    domain_loss_visual) / 3
        return adv_loss
