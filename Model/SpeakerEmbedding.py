import torch
import torch.nn as nn


class SpeakerEmbedding(nn.Module):

    def __init__(self, model_dim, num_speakers, text_dim, audio_dim,
                 visual_dim, if_shared_speaker_embedding):
        super(SpeakerEmbedding, self).__init__()
        self.if_shared_speaker_embedding = if_shared_speaker_embedding

        print("self.if_shared_speaker_embedding",
              self.if_shared_speaker_embedding)
        # Speaker Embedding By speaker_masks (one hot)
        if if_shared_speaker_embedding:
            self.shared_speaker_embedding = nn.Embedding(
                num_embeddings=num_speakers, embedding_dim=model_dim)
        else:
            # self.text_speaker_embedding = nn.Embedding(
            #     num_embeddings=num_speakers, embedding_dim=text_dim)
            # self.audio_speaker_embedding = nn.Embedding(
            #     num_embeddings=num_speakers, embedding_dim=audio_dim)
            # self.visual_speaker_embedding = nn.Embedding(
            #     num_embeddings=num_speakers, embedding_dim=visual_dim)
            self.text_speaker_embedding = nn.Embedding(
                num_embeddings=num_speakers, embedding_dim=model_dim)
            self.audio_speaker_embedding = nn.Embedding(
                num_embeddings=num_speakers, embedding_dim=model_dim)
            self.visual_speaker_embedding = nn.Embedding(
                num_embeddings=num_speakers, embedding_dim=model_dim)

    def forward(self, speaker_masks, utterance_masks):
        batch_size, seq_len, _ = speaker_masks.size()
        if utterance_masks is not None and False:
            print('speaker_masks.size()', speaker_masks.size())
            print('utterance_masks.size()', utterance_masks.size())
            assert utterance_masks.size() == (
                batch_size, seq_len), "Utterance masks shape mismatch"
        # one-hot to speaker_ids
        speaker_ids = torch.argmax(speaker_masks, dim=-1)

        if self.if_shared_speaker_embedding:
            speaker_embed = self.shared_speaker_embedding(speaker_ids)
            if utterance_masks is not None:
                speaker_embed = speaker_embed * utterance_masks.unsqueeze(-1)
            # print("Use shared speaker embedding")
            # print('text_features.shape', text_features.shape)
            # print('speaker_embed.shape', speaker_embed.shape)

            return speaker_embed.transpose(0, 1)
        else:
            # Speaker Embedding
            text_speaker_embed = self.text_speaker_embedding(speaker_ids)
            audio_speaker_embed = self.audio_speaker_embedding(speaker_ids)
            visual_speaker_embed = self.visual_speaker_embedding(speaker_ids)
            if utterance_masks is not None:
                text_speaker_embed = text_speaker_embed * utterance_masks.unsqueeze(
                    -1)
                audio_speaker_embed = audio_speaker_embed * utterance_masks.unsqueeze(
                    -1)
                visual_speaker_embed = visual_speaker_embed * utterance_masks.unsqueeze(
                    -1)
            # print("Use unique speaker embedding")
            # print('text_features.shape', texts.shape)
            # print('speaker_embed.shape', text_speaker_embed.shape)

            return text_speaker_embed.transpose(
                0, 1), audio_speaker_embed.transpose(
                    0, 1), visual_speaker_embed.transpose(0, 1)
