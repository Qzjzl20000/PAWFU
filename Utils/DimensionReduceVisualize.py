import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import os


class DimensionReduceVisualize:

    def __init__(self, dataset, train_time, data_name):
        self.dataset = dataset
        self.train_time = train_time
        self.data_name = data_name
        self.visual_data = []
        self.labels = []

    def add_data(
        self,
        data,
        labels,
    ):
        """
        添加要可视化的数据和标签。
        
        :param data: 特征或logits (batch_size * seq_len, model_dim or num_class)
        :param labels: 对应的标签 (batch_size * seq_len)
        """

        self.visual_data = data
        self.labels = labels

    def visualize(self,
                  method='t-sne',
                  n_components=2,
                  epoch=None,
                  ifBest=False):
        """
        执行降维和可视化。
        
        :param method: 降维方法 ('t-sne' or 'umap')
        :param n_components: 降维后的维度数 (2 或 3)
        :param epoch: 当前epoch编号，用于文件命名
        :param ifBest: 是否为最佳模型的结果
        """
        all_data = np.concatenate(self.visual_data, axis=0)
        all_labels = np.concatenate(self.labels, axis=0)

        print(f'{method.upper()} Visualization on {self.dataset}')
        print('Data shape:', all_data.shape)
        print('Labels shape:', all_labels.shape)

        if method == 't-sne':
            reducer = TSNE(n_components=n_components,
                           perplexity=30,
                           learning_rate=200,
                           random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("Unsupported dimensionality reduction method")

        embedded_data = reducer.fit_transform(all_data)

        self._plot(embedded_data, all_labels, method, n_components, epoch,
                   ifBest)

    def _plot(self, embedded_data, labels, method, n_components, epoch,
              ifBest):
        """
        内部方法，用于绘制降维后的数据。
        """
        if n_components == 2:
            fig = plt.figure(figsize=(8, 6))
            scatter = plt.scatter(embedded_data[:, 0],
                                  embedded_data[:, 1],
                                  c=labels,
                                  cmap='viridis')
            plt.colorbar(scatter, label='Label')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title(f'{method.upper()} Visualization 2D on {self.dataset}')
        elif n_components == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embedded_data[:, 0],
                                 embedded_data[:, 1],
                                 embedded_data[:, 2],
                                 c=labels,
                                 cmap='viridis')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(
                f'{method.upper()} Visualization 3D on {self.dataset}')
            fig.colorbar(scatter, ax=ax, label='Label')

        save_dir = f'./SaveModel/{self.dataset}/'
        os.makedirs(save_dir, exist_ok=True)

        file_prefix = f"{self.train_time}_{self.data_name}_{method.upper()}_{n_components}d_on_{self.dataset}_in_epoch_{epoch}"
        if ifBest:
            file_prefix += "_best"

        fig.savefig(os.path.join(save_dir, f"{file_prefix}.png"))
        plt.close(fig)
        print(f"Saved {method.upper()} figure in epoch: {epoch}")


# def T_SNE_visual(dataset,
#                  DR_visual_logits_logit_list,
#                  DR_visual_logits_label_list,
#                  epoch,
#                  train_time,
#                  ifBest=False):

#     TSNE_visual_logit_np = np.concatenate(DR_visual_logits_logit_list, axis=0)
#     TSNE_visual_label_np = np.concatenate(DR_visual_logits_label_list, axis=0)

#     print('T_SNE_Visual_list np.shape')
#     print('logit_list:', TSNE_visual_logit_np.shape)
#     print('label_list:', TSNE_visual_label_np.shape)

#     # TSNE - 2d
#     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
#     embedded_data = tsne.fit_transform(TSNE_visual_logit_np)
#     # distances = pairwise_distances(embedded_data, metric='euclidean')
#     # plt.scatter(embedded_data, TSNE_visual_label_np, c=mean_distances)
#     plt.scatter(embedded_data[:, 0],
#                 embedded_data[:, 1],
#                 c=TSNE_visual_label_np)
#     plt.colorbar()
#     plt.xlabel('featrues')
#     plt.ylabel('label')
#     plt.title(f't-SNE Visualization 2d on {dataset}')
#     # plt.colorbar(label='Mean Distance')
#     # plt.show()
#     if ifBest:
#         plt.savefig(
#             f'./SaveModel/{dataset}/{train_time}_TSNE_2d_on_{dataset}_in_epoch_{epoch}_best.png'
#         )
#         # plt.savefig(
#         #     f'./SaveModel/{dataset}/{train_time}_TSNE_2d_on_{dataset}_in_epoch_{epoch}_best.pdf',
#         #     dpi=300,
#         #     format='pdf')
#         print(f"Save Best TSNE fig 2d in epoch:{epoch}")
#     else:
#         plt.savefig(
#             f'./SaveModel/{dataset}/TSNE_2d_on_{dataset}_in_epoch_{epoch}.png')
#         # plt.savefig(f'./SaveModel/{dataset}/TSNE_2d_on_{dataset}_in_epoch_{epoch}.pdf',
#         #             dpi=300,
#         #             format='pdf')
#         print(f"Save TSNE fig 2d in epoch:{epoch}")
#     plt.close()

#     # TSNE - 3d
#     tsne = TSNE(n_components=3, random_state=2024)
#     X_tsne = tsne.fit_transform(TSNE_visual_logit_np)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X_tsne[:, 0],
#                X_tsne[:, 1],
#                X_tsne[:, 2],
#                c=TSNE_visual_label_np)
#     ax.set_xlabel('Dimension 1')
#     ax.set_ylabel('Dimension 2')
#     ax.set_zlabel('Dimension 3')
#     ax.set_title(f't-SNE Visualization 3d on {dataset}')
#     # plt.show()
#     if ifBest:
#         plt.savefig(
#             f'./SaveModel/{dataset}/{train_time}_TSNE_3d_on_{dataset}_in_epoch_{epoch}_best.png'
#         )
#         # plt.savefig(
#         #     f'./SaveModel/{dataset}/{train_time}_TSNE_3d_on_{dataset}_in_epoch_{epoch}_best.pdf',
#         #     dpi=300,
#         #     format='pdf')
#         print(f"Save Best TSNE fig 3d in epoch:{epoch}")
#     else:
#         plt.savefig(
#             f'./SaveModel/{dataset}/TSNE_3d_on_{dataset}_in_epoch_{epoch}.png')
#         # plt.savefig(f'./SaveModel/{dataset}/TSNE_3d_on_{dataset}_in_epoch_{epoch}.pdf',
#         #             dpi=300,
#         #             format='pdf')
#         print(f"Save TSNE fig 3d in epoch:{epoch}")
#     plt.close()
#     return
