import torch.nn as nn


class AdaptiveClassifier(nn.Module):

    def __init__(self,
                 model_dim,
                 input_dim,
                 num_classes,
                 dropout_rate,
                 layer_config=None):
        super(AdaptiveClassifier, self).__init__()

        # 默认配置
        if layer_config is None:
            num = input_dim // model_dim
            if num >= 9:
                layer_config = [(input_dim, input_dim // 3),
                                (input_dim // 3, input_dim // 9),
                                (input_dim // 9, num_classes)]
            elif num >= 6:
                layer_config = [(input_dim, input_dim // 2),
                                (input_dim // 2, input_dim // 6),
                                (input_dim // 6, num_classes)]
            elif num >= 3:
                layer_config = [(input_dim, input_dim // 3),
                                (input_dim // 3, input_dim // 6),
                                (input_dim // 6, num_classes)]
            else:
                layer_config = [(input_dim, input_dim // 2),
                                (input_dim // 2, num_classes)]
        else:
            layer_config = layer_config
            # layer_config = [(input_dim, 512), (512, 256), (256, 128),
            #                 (128, num_classes)]
        layers = []
        for in_features, out_features in layer_config[:-1]:
            layers.extend([
                nn.LayerNorm(in_features),
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
        # 最后一层不需要激活函数和dropout
        final_in, final_out = layer_config[-1]
        layers.append(nn.Linear(final_in, final_out))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
