import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Utils.GatedFusion import AdaptiveGatedFusion
from Utils.AdaptiveClassifier import AdaptiveClassifier
from Model.SingleModality import SingleModality
from Model.DualModality import DualModality
from Model.UncertaintyEstimate import UncertaintyEstimate
from Model.ModalityAlignment import ModalityAlignment
from Model.DynamicWeightedFusion import DynamicWeightedFusion
from Model.CrossModalAttentionFusion import CrossModalAttentionFusion
from Model.SpeakerEmbedding import SpeakerEmbedding
from Model.DualModality import DualModalityEnhanced
from Model.CrossFusion import CrossFusion
from Model.SingleModalityClassifierSelfDistill import SingleModalityClassifierSelfDistill
from Model.PreAlignment import PreAlignment
from Model.AlignModule import AlignModule
from Model.DomainAdversarialModule import DomainAdversarialModule


class PAWFU(nn.Module):

    def __init__(self,
                 device,
                 args,
                 text_dim=768,
                 audio_dim=512,
                 visual_dim=1000,
                 num_classes=7,
                 num_speakers=9):
        super(PAWFU, self).__init__()
        self.align_module = AlignModule(text_dim, audio_dim, visual_dim,
                                        args.model_dim, args.hidden_dim)
        self.domain_adv_module = DomainAdversarialModule(
            args.model_dim, args.hidden_dim)

        self.main_task_module = MainTaskModule(device, args, text_dim,
                                               audio_dim, visual_dim,
                                               num_classes, num_speakers)

    def forward(self,
                texts,
                audios,
                visuals,
                speaker_masks,
                utterance_masks,
                labels,
                valid_indices,
                if_optimizer_adv=False):
        if if_optimizer_adv:
            text_features, audio_features, visual_features = self.main_task_module(
                texts,
                audios,
                visuals,
                speaker_masks,
                utterance_masks,
                labels,
                valid_indices,
                if_optimizer_adv=True)
            return self.domain_adv_module.domain_adversarial_loss(
                text_features, audio_features, visual_features)
        else:
            return self.main_task_module(texts,
                                         audios,
                                         visuals,
                                         speaker_masks,
                                         utterance_masks,
                                         labels,
                                         valid_indices,
                                         if_optimizer_adv=False)


class MainTaskModule(nn.Module):

    def __init__(self,
                 device,
                 args,
                 text_dim=768,
                 audio_dim=512,
                 visual_dim=1000,
                 num_classes=7,
                 num_speakers=9):
        super(MainTaskModule, self).__init__()
        self.device = device
        self._init_hyperparameters(args)
        self._init_speaker_embedding(num_speakers, text_dim, audio_dim,
                                     visual_dim)
        self._init_modalities(args.dataset, text_dim, audio_dim, visual_dim,
                              num_classes, num_speakers, device)
        self._init_self_distillation(num_classes)
        self._init_cross_attention_fusion()
        self._init_uncertainty_module()
        self._init_dynamic_adaptive_fusion_classifier(num_classes)

    def _init_hyperparameters(self, args):
        self.modality = args.modality
        self.model_dim = args.model_dim
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate
        self.dual_fusion_layer_num = args.dual_fusion_layer_num
        self.if_use_speaker_embedding = args.if_use_speaker_embedding
        self.speaker_embedding_ratio = args.speaker_embedding_ratio
        self.if_shared_speaker_embedding = args.if_shared_speaker_embedding
        self.if_use_cross_attention_fusion = False
        self.if_use_concat_classifier = True
        self.self_distill_num = args.self_distill_num
        self.if_reset_weight_to_one = args.if_use_self_distill == 0
        self.if_use_weight_entropy_temp = args.if_use_weight_entropy_temp != 0.0
        self.base_temp = args.if_use_weight_entropy_temp if self.if_use_weight_entropy_temp else nn.Parameter(
            torch.ones(1))
        self.if_use_uncertainty_to_temperature = True

    def _init_speaker_embedding(self, num_speakers, text_dim, audio_dim,
                                visual_dim):
        if self.if_use_speaker_embedding:
            self.speaker_embedding = SpeakerEmbedding(
                self.model_dim, num_speakers, text_dim, audio_dim, visual_dim,
                self.if_shared_speaker_embedding)

    def _init_modalities(self, dataset, text_dim, audio_dim, visual_dim,
                         num_classes, num_speakers, device):
        self.single_modality = SingleModality(dataset, text_dim, audio_dim,
                                              visual_dim, self.model_dim,
                                              self.dropout_rate, num_classes,
                                              num_speakers, device)
        if self.dual_fusion_layer_num > 0:
            self.dual_modality = DualModality(self.model_dim, self.num_heads,
                                              self.hidden_dim,
                                              self.dual_fusion_layer_num,
                                              self.dropout_rate)

    def _init_self_distillation(self, num_classes):
        if self.self_distill_num > 0:
            self.sd_dim_times = 2 if self.dual_fusion_layer_num > 0 else 1
            self.distill_modules = {}
            combinations = [("T", "A"), ("T", "V"), ("A", "V")]
            if self.self_distill_num >= 6:
                combinations += [("A", "T"), ("V", "T"), ("V", "A")]
            if self.self_distill_num == 9:
                combinations += [("T", "T"), ("A", "A"), ("V", "V")]

            for combo in combinations:
                self.distill_modules["_".join(
                    combo)] = SingleModalityClassifierSelfDistill(
                        self.model_dim, self.sd_dim_times, num_classes,
                        self.dropout_rate)

    def _init_cross_attention_fusion(self):
        if self.if_use_cross_attention_fusion:
            self.caf_dim_times = 6 if self.if_use_concat_classifier else 1
            self.cross_fusion = CrossFusion(self.model_dim, self.caf_dim_times,
                                            self.num_heads, self.hidden_dim,
                                            self.dropout_rate)

    def _init_uncertainty_module(self):
        self.uncertainty_input_dim = self.self_distill_num
        self.uncertainty_output_dim = self.self_distill_num
        self.uncertainty_generator = nn.Sequential(
            nn.Linear(self.uncertainty_input_dim, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(self.dropout_rate), nn.Linear(128, 128),
            nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.uncertainty_output_dim))

    def _init_dynamic_adaptive_fusion_classifier(self, num_classes):
        layer_config = [
            (self.model_dim *
             (self.self_distill_num if self.if_use_concat_classifier else 1),
             self.model_dim * 3), (self.model_dim * 3, self.model_dim),
            (self.model_dim, self.model_dim // 2),
            (self.model_dim // 2, num_classes)
        ]
        self.dynamic_adaptive_fusion_classifier = AdaptiveClassifier(
            self.model_dim,
            self.model_dim *
            (self.self_distill_num if self.if_use_concat_classifier else 1),
            num_classes,
            self.dropout_rate,
            layer_config=layer_config)

    def forward(self, args, e, text_features, audio_features, visual_features,
                speaker_masks, utterance_masks, labels, valid_indices):
        batch_size, seq_len = text_features.size(0), text_features.size(1)

        if args.if_use_dialogue_rnn:
            text_features, audio_features, visual_features = self._apply_speaker_embedding(
                text_features, audio_features, visual_features, speaker_masks,
                utterance_masks)

        stats = None
        dual_features = self._apply_dual_modality(text_features,
                                                  audio_features,
                                                  visual_features)
        fusion_T, fusion_A, fusion_V = self._compute_fusion_features(
            dual_features)

        if self.self_distill_num > 0:
            final_features, total_teacher_training_CE_loss, total_weight_MSE_loss, weight_list, teacher_logits_list, uncertainty_record = (
                self._apply_self_distillation(fusion_T, fusion_A, fusion_V,
                                              dual_features, labels,
                                              valid_indices, args))
        else:
            final_features = self._compute_final_features(
                fusion_T, fusion_A, fusion_V, dual_features)
            total_teacher_training_CE_loss, total_weight_MSE_loss, teacher_logits_list, uncertainty_record = None, None, None, None

        if self.if_use_cross_attention_fusion:
            final_features = self.cross_fusion(final_features)

        logits_DAF = self.dynamic_adaptive_fusion_classifier(
            final_features.reshape(-1, final_features.size(-1)))

        return logits_DAF, total_teacher_training_CE_loss, total_weight_MSE_loss, weight_list, teacher_logits_list, stats, uncertainty_record

    def _apply_speaker_embedding(self, text_features, audio_features,
                                 visual_features, speaker_masks,
                                 utterance_masks):
        if not self.if_use_speaker_embedding:
            return self.single_modality(text_features, audio_features,
                                        visual_features,
                                        speaker_masks.transpose(0, 1),
                                        utterance_masks)

        speaker_embed = self.speaker_embedding(speaker_masks, utterance_masks)
        if self.if_shared_speaker_embedding:
            return self.single_modality(
                text_features + self.speaker_embedding_ratio * speaker_embed,
                audio_features + self.speaker_embedding_ratio * speaker_embed,
                visual_features + self.speaker_embedding_ratio * speaker_embed,
                speaker_masks.transpose(0, 1), utterance_masks)

        text_speaker_embed, audio_speaker_embed, visual_speaker_embed = speaker_embed
        return self.single_modality(
            text_features + self.speaker_embedding_ratio * text_speaker_embed,
            audio_features +
            self.speaker_embedding_ratio * audio_speaker_embed,
            visual_features +
            self.speaker_embedding_ratio * visual_speaker_embed,
            speaker_masks.transpose(0, 1), utterance_masks)

    def _apply_dual_modality(self, text_features, audio_features,
                             visual_features):
        return self.dual_modality(
            text_features, audio_features,
            visual_features) if self.dual_fusion_layer_num > 0 else (
                text_features, audio_features, visual_features, text_features,
                audio_features, text_features, visual_features, audio_features,
                visual_features)

    def _compute_fusion_features(self, dual_features):
        ta_features, at_features, tv_features, vt_features, av_features, va_features = dual_features[
            3:]
        return torch.cat((ta_features, tv_features), dim=-1), torch.cat(
            (at_features, av_features), dim=-1), torch.cat(
                (vt_features, va_features), dim=-1)

    def _apply_self_distillation(self, fusion_T, fusion_A, fusion_V,
                                 dual_features, labels, valid_indices, args):
        distill_methods = {
            3: self._self_distill_three,
            6: self._self_distill_six,
            9: self._self_distill_nine
        }
        return distill_methods.get(
            self.self_distill_num, lambda *_:
            (None, None, None, None, None, None))(fusion_T, fusion_A, fusion_V,
                                                  dual_features, labels,
                                                  valid_indices, args)

    def _compute_final_features(self, fusion_T, fusion_A, fusion_V,
                                dual_features):
        return torch.cat((fusion_T, fusion_A, fusion_V), dim=-1) if self.if_use_concat_classifier else \
            torch.mean(torch.stack(dual_features[3:], dim=-1), dim=-1)

    def reset_tensor_list_to_one_weight(self, tensor_list):
        return [torch.ones_like(tensor) for tensor in tensor_list]

    def compute_normalized_gini(self, prob):
        gini = 1 - (prob**2).sum(dim=-1, keepdim=True)
        max_gini = 1 - 1 / prob.size(-1)
        return gini / max_gini

    def compute_entropy(self, probabilities, max_entropy):
        entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(
            dim=-1, keepdim=True)
        return entropy / max_entropy

    def uncertainty_entropy_guided_temperature(self, weight_stack,
                                               logits_stack):
        batch_size, seq_len, num_branches, num_classes = logits_stack.size()

        if not hasattr(self, 'temp_sensitive'):
            self.temp_sensitive = nn.Parameter(
                torch.tensor(1.0 * self.base_temp)).to(weight_stack.device)

        H_max_branch = math.log(num_branches)
        H_max_class = math.log(num_classes)

        prob_weights = F.softmax(weight_stack.squeeze(-1), dim=-1)
        weight_entropy = self.compute_entropy(prob_weights, H_max_branch)

        logits_prob = F.softmax(logits_stack, dim=-1)
        logits_entropy = self.compute_entropy(logits_prob,
                                              H_max_class).squeeze(-1)

        input_entropies = [logits_entropy]
        if self.if_cat_weight_entropy:
            input_entropies.append(weight_entropy)
        if self.if_cat_logits_list:
            input_entropies.append(logits_stack.view(batch_size, seq_len, -1))
        if self.if_cat_weights_list:
            input_entropies.append(prob_weights.view(batch_size, seq_len, -1))

        input_entropies = torch.cat(input_entropies, dim=-1)
        uncertainty = self.uncertainty_generator(input_entropies)

        temperature = torch.exp(
            self.temp_sensitive * uncertainty
        ) if self.uncertainty_output_dim == 1 or self.if_use_uncertainty_to_temperature else torch.ones(
            batch_size, seq_len, 1).to(weight_stack.device)

        return uncertainty, temperature

    def softmax_weight_list(self,
                            teacher_logits_list,
                            weight_list,
                            valid_indices,
                            if_count_weight_ratio_or_num=False,
                            if_use_weight_entropy_temp=False):
        stats = {}
        weight_num = len(weight_list)
        batch_size, seq_len = weight_list[0].shape[:2]

        weight_stack = torch.stack(weight_list, dim=2)
        logits_stack = torch.stack(teacher_logits_list, dim=2)

        if if_use_weight_entropy_temp:
            uncertainty, temperature = self.uncertainty_entropy_guided_temperature(
                weight_stack, logits_stack)
            scaling_factor = temperature.unsqueeze(
                -1) if self.uncertainty_output_dim == 1 else (
                    temperature
                    if self.if_use_uncertainty_to_temperature else uncertainty)
            weight_softmax = F.softmax(
                weight_stack.squeeze(-1) / scaling_factor,
                dim=2).unsqueeze(-1) * weight_num
        else:
            weight_softmax = F.softmax(weight_stack, dim=2) * weight_num

        softmax_weights_list = torch.unbind(weight_softmax, dim=-2)

        if if_count_weight_ratio_or_num:
            weight_record = weight_softmax.squeeze(-1)
            rank_tensor = torch.argsort(torch.argsort(weight_record, dim=-1),
                                        dim=-1)

            if valid_indices is not None:
                rank_tensor = rank_tensor.view(-1, weight_num)[valid_indices]
                weight_record = weight_record.view(-1,
                                                   weight_num)[valid_indices]
                uncertainty = uncertainty.view(
                    -1, uncertainty.size(-1))[valid_indices]
                temperature = temperature.view(
                    -1, temperature.size(-1))[valid_indices]

            stats['rank'] = rank_tensor
            stats['weight'] = weight_record
            uncertainty_record = torch.cat((uncertainty, temperature), dim=-1)
        else:
            uncertainty_record = None

        return softmax_weights_list, stats, uncertainty_record

    def analyze_ranks(self, rank_tensor):
        avg_ranks = rank_tensor.float().mean(dim=(0, 1)).tolist()
        print("Average Ranks:", avg_ranks)
