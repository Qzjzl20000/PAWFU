import torch
import torch.nn.functional as F


class UncertaintyEstimate:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_kl_divergence(self, logits1, logits2, reduction='none'):

        # F.kl_div need to ensure logits1 is log-probabilities，logits2 is probabilities
        log_prob_dist1 = F.log_softmax(logits1, dim=-1)
        prob_dist2 = F.softmax(logits2, dim=-1)

        # KL loss
        kl_divergence = F.kl_div(
            log_prob_dist1, prob_dist2,
            reduction='none').sum(dim=-1, keepdim=True)  # (batch * seq_len,1)

        # if reduction == 'batchmean':
        #     kl_divergence = kl_divergence.mean()
        # elif reduction == 'sum':
        #     kl_divergence = kl_divergence.sum()

        return kl_divergence  # (batch * seq_len,1)
