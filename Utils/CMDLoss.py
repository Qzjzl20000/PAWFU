import torch
import torch.nn as nn


class CMDLoss(nn.Module):

    def __init__(self, kernel='rbf', gamma=1.0):
        """
        CMD（Conditional Maximum Mean Discrepancy） Loss 实现
        :param kernel: 核函数类型 ('rbf' or 'linear')
        :param gamma: RBF 核函数的超参数
        """
        super(CMDLoss, self).__init__()
        self.kernel = kernel
        self.gamma = gamma

    def compute_kernel(self, X, Y):
        """
        计算 X 和 Y 之间的核矩阵
        :param X: Tensor, shape (batch_size * seq_len, model_dim)
        :param Y: Tensor, shape (batch_size * seq_len, model_dim)
        :return: Kernel matrix of shape (batch_size * seq_len, batch_size * seq_len)
        """
        if self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        else:
            raise ValueError("Unsupported kernel type")

    def rbf_kernel(self, X, Y):
        """
        RBF 核函数
        :param X: Tensor, shape (batch_size * seq_len, model_dim)
        :param Y: Tensor, shape (batch_size * seq_len, model_dim)
        :return: Kernel matrix of shape (batch_size * seq_len, batch_size * seq_len)
        """
        X_norm = torch.sum(X**2, dim=1).view(-1, 1)  # ||X||^2
        Y_norm = torch.sum(Y**2, dim=1).view(-1, 1)  # ||Y||^2
        dist = X_norm + Y_norm.T - 2.0 * torch.mm(X, Y.T)  # ||X - Y||^2
        kernel = torch.exp(-self.gamma * dist)  # RBF kernel
        return kernel

    def linear_kernel(self, X, Y):
        """
        线性核函数
        :param X: Tensor, shape (batch_size * seq_len, model_dim)
        :param Y: Tensor, shape (batch_size * seq_len, model_dim)
        :return: Kernel matrix of shape (batch_size * seq_len, batch_size * seq_len)
        """
        return torch.mm(X, Y.T)

    def forward(self, X, Y, condition=None):
        """
        计算 CMD 损失
        :param X: 模态 X 的特征 (batch_size, seq_len, model_dim)
        :param Y: 模态 Y 的特征 (batch_size, seq_len, model_dim)
        :param condition: 条件变量 (batch_size, seq_len)，如类别标签或说话者 ID
        :return: CMD 损失
        """

        # Reshape to (batch_size * seq_len, model_dim)
        batch_size, seq_len, model_dim = X.size()
        X = X.view(batch_size * seq_len, model_dim)
        Y = Y.view(batch_size * seq_len, model_dim)
        if condition is not None:
            # Flatten the condition variable if it has a sequence length dimension
            if len(condition.shape) > 1:
                condition = condition.view(-1)

            unique_conditions = torch.unique(condition)  # 找到所有唯一条件
            total_loss = 0.0

            for c in unique_conditions:
                mask = condition == c  # 筛选当前条件的数据
                X_c = X[mask]
                Y_c = Y[mask]

                if X_c.size(0) == 0 or Y_c.size(0) == 0:
                    continue  # 如果没有匹配的条件，则跳过

                # 核函数计算
                K_XX = self.compute_kernel(X_c, X_c)
                K_YY = self.compute_kernel(Y_c, Y_c)
                K_XY = self.compute_kernel(X_c, Y_c)

                # CMD 损失
                cmd_loss = torch.mean(K_XX) + torch.mean(
                    K_YY) - 2 * torch.mean(K_XY)
                total_loss += cmd_loss

            # 防止除以零的情况
            num_unique_conditions = len(unique_conditions)
            loss = total_loss / max(num_unique_conditions, 1)  # 归一化损失
        else:
            # 核函数计算
            K_XX = self.compute_kernel(X, X)
            K_YY = self.compute_kernel(Y, Y)
            K_XY = self.compute_kernel(X, Y)

            # CMD 损失
            loss = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)

        return loss
