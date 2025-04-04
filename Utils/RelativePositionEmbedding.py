import torch
import torch.nn as nn


class RelativePositionEmbedding(nn.Module):

    def __init__(self, d_model, max_relative_positions):
        super(RelativePositionEmbedding, self).__init__()
        self.max_relative_positions = max_relative_positions
        vocab_size = max_relative_positions * 2 + 1
        self.embedding_table = nn.Parameter(torch.Tensor(vocab_size, d_model))
        nn.init.xavier_uniform_(self.embedding_table)

    def forward(self, length):
        if length > self.max_relative_positions:
            raise ValueError(
                "Sequence length exceeds the maximum relative position length")

        # Generate relative positions matrix
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(distance_mat,
                                           min=-self.max_relative_positions,
                                           max=self.max_relative_positions)
        final_mat = distance_mat_clipped + self.max_relative_positions

        embeddings = self.embedding_table[final_mat]
        return embeddings
