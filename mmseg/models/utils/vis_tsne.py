import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from skimage import exposure
import cv2


# 对样本进行预处理并画图
def plot_embedding(data, label, name):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 遍历所有样本
    plt.figure()
    # for i in range(data.shape[0]):
    # 在图中为每个数据点画出标签
    # plt.text(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 2),
    #          fontdict={'weight': 'bold', 'size': 7})
    plt.scatter(data[label == 0, 0], data[label == 0, 1], s=3, color='blue', label='background', alpha=0.2)
    plt.scatter(data[label == 1, 0], data[label == 1, 1], s=3, color='green', label='demolished', alpha=0.2)
    plt.scatter(data[label == 2, 0], data[label == 2, 1], s=3, color='red', label='newly built', alpha=0.2)

    #plt.legend()
    plt.axis('off')
    plt.savefig(name, dpi=300, format='png')

def plot_embedding_multimodal(data, label, name):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 遍历所有样本
    plt.figure()
    # for i in range(data.shape[0]):
    # 在图中为每个数据点画出标签
    # plt.text(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 2),
    #          fontdict={'weight': 'bold', 'size': 7})
    plt.scatter(data[label == 0, 0], data[label == 0, 1], s=3, color='blue', label='background', alpha=0.2)
    plt.scatter(data[label == 1, 0], data[label == 1, 1], s=3, color='red', label='demolished', alpha=0.2)

    #plt.legend()
    plt.axis('off')
    plt.savefig(name, dpi=300, format='png')
# 主函数，执行t-SNE降维
def visualize_tsne_multimodal(data, name, total=10000):
    data = data.detach().cpu().numpy()
    C, LL = data.shape
    L = LL//2

    modal_1 = data[:, :L].transpose(1, 0)
    modal_2 = data[:, L:].transpose(1, 0)

    idx = np.arange(0, L)
    np.random.shuffle(idx)

    label = np.zeros(total)
    label[:total//2] = 1
    data = np.concatenate([modal_1[idx][:total//2],modal_2[idx][:total//2]], axis=0)
    print('Count label!=0: %d ' % label.sum())
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    result = ts.fit_transform(data)
    # 调用函数，绘制图像
    # abel is used for coloring
    # result -> [N,2] means [N,(x,y)]
    plot_embedding_multimodal(result, label, name)

# 主函数，执行t-SNE降维
def visualize_tsne(data, label, name, total=10000):
    data = data.detach().cpu().numpy()[0]
    C, H, W = data.shape

    label = label[0].detach().cpu().numpy()
    label = cv2.resize(label, (H, W)).reshape(-1)
    #label[label != 0] = 1
    data = data.transpose(1, 2, 0)
    data = np.reshape(data, [data.shape[0] ** 2, data.shape[-1]])
    idx = np.arange(0, H ** 2)

    idx = list(idx[label == 0][:total // 2]) + list(idx[label != 0][:total // 2])
    np.random.shuffle(idx)
    data = data[idx[:total], ::]
    label = label[idx[:total]]
    print('Count label!=0: %d ' % label.sum())
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    result = ts.fit_transform(data)
    # 调用函数，绘制图像
    # abel is used for coloring
    # result -> [N,2] means [N,(x,y)]
    plot_embedding(result, label, name)
    # return result, label
