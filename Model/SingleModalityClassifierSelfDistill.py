import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleModalityClassifierSelfDistill(nn.Module):

    def __init__(self, model_dim, dim_times, num_class, dropout_rate=0.1):
        super(SingleModalityClassifierSelfDistill, self).__init__()
        self.model_dim = model_dim
        self.dim_times = dim_times
        self.feature_dim = model_dim * dim_times
        self.num_class = num_class
        self.teacher_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 4, num_class))
        self.student_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(self.feature_dim // 4, 1))
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()

    def forward(self, feature_input, labels, valid_indices):
        batch_size, seq_len, _ = feature_input.size()

        # Teacher MLP Predict and Train
        teacher_logits = self.teacher_mlp(
            feature_input)  #(batch_size, seq_len,num_class)

        # Classifier CE Loss
        logit_size = teacher_logits.size(-1)
        teacher_logits_expand = teacher_logits.view(-1, logit_size)
        teacher_logits_expand = teacher_logits_expand[valid_indices]
        teacher_training_CE_loss = self.CE_loss(
            F.softmax(teacher_logits_expand), labels)
        # print(
        #     f"teacher_logits.size():{teacher_logits.size()}, labels.size():{labels.size()}"
        # )

        # Student MLP Predict and Train
        teacher_predict = torch.gather(
            F.softmax(teacher_logits_expand) * logit_size, -1,
            labels.unsqueeze(-1))
        # teacher_predict = torch.gather(teacher_logits_expand, -1,
        #                                labels.unsqueeze(-1))
        weight_param = self.student_mlp(feature_input)
        student_predict = weight_param.clone()

        student_predict = student_predict.view(-1)
        student_predict = student_predict[valid_indices]

        weight_MSE_loss = self.MSE_loss(student_predict, teacher_predict)

        # print(
        #     f"student_predict:{student_predict.size()}, teacher_training_CE_loss:{teacher_training_CE_loss.size()}, weight_MSE_loss:{weight_MSE_loss.size()}"
        # )

        return weight_param, teacher_training_CE_loss, weight_MSE_loss, teacher_logits

    def multimodal_guided_KL_loss(DAF_mm_logits,
                                  SD_single_teacher_logits_list,
                                  weight_list,
                                  valid_indices,
                                  temperature=1.0,
                                  if_use_balanced_KL=True):
        # KL(Q∣∣P):= ∑Q(x)log(Q(x)/P(x))
        # P Guided Q
        # Q: SD_single_teacher_logits_list
        # P: DAF_mm_logits
        MMG_KL_loss = 0.0
        MMG_KL_list = []

        class_num = DAF_mm_logits.size(-1)

        new_teacher_logits_list = []
        for logits in SD_single_teacher_logits_list:
            logits = logits.view(-1, class_num)
            logits = logits[valid_indices]
            new_teacher_logits_list.append(logits)
        SD_single_teacher_logits_list = new_teacher_logits_list

        new_weight_list = []
        for weight in weight_list:
            weight = weight.view(-1, 1)
            weight = weight[valid_indices]
            new_weight_list.append(weight)
        weight_list = new_weight_list

        if if_use_balanced_KL:

            for SD_single_teacher_logits, single_weight in zip(
                    SD_single_teacher_logits_list, weight_list):

                # print("SD_single_teacher_logits.size()",
                #       SD_single_teacher_logits.size())
                # print('single_weight.size()', single_weight.size())

                SD_single_teacher_probs = F.log_softmax(
                    SD_single_teacher_logits / temperature, dim=-1)
                DAF_mm_probs = F.softmax(DAF_mm_logits / temperature, dim=-1)
                KL = F.kl_div(SD_single_teacher_probs,
                              DAF_mm_probs,
                              reduction='none').sum(dim=-1).unsqueeze(-1)
                KL = (KL * single_weight).mean()

                # KL *= (temperature**2)

                MMG_KL_list.append(KL)
        else:

            for SD_single_teacher_logits in SD_single_teacher_logits_list:
                SD_single_teacher_probs = F.log_softmax(
                    SD_single_teacher_logits / temperature, dim=-1)
                DAF_mm_probs = F.softmax(DAF_mm_logits / temperature, dim=-1)
                KL = F.kl_div(SD_single_teacher_probs,
                              DAF_mm_probs,
                              reduction='batchmean')

                # KL *= (temperature**2)

                MMG_KL_list.append(KL)

        MMG_KL_loss = torch.stack(MMG_KL_list, dim=-1).mean()

        return MMG_KL_loss, MMG_KL_list

    def weight_balance_reg_loss(weight_list):
        # weight size (batch_size , seq_len, 1)
        mean_weight = 1.0
        feature_num = len(weight_list)
        weight_stack = torch.stack(weight_list, dim=-1).reshape(
            -1, feature_num)  # (batch_size * seq_len,feature_num)
        reg_loss_stack = ((weight_stack - mean_weight)**2).mean(
            dim=-1)  # (batch_size * seq_len)
        reg_loss = reg_loss_stack.mean()  # (1)
        return reg_loss
