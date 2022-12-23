import pickle

import dgl
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier

from mlp_cs.similar import similar_str
import csv
import os
import openpyxl as openpyxl
import numpy as np



def AdjustedCosine(dataA,dataB,avg):
    sumData = (dataA - avg) * (dataB - avg).T # 若列为向量则为 dataA.T * dataB
    denom = np.linalg.norm(dataA - avg) * np.linalg.norm(dataB - avg)
    return 0.5 + 0.5 * (sumData / denom)


def feature_selection(matrix, labels, train_ind, fnum):
    os.chdir(r'E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats')
    wb = openpyxl.Workbook()
    sheet = wb.active
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection 

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=1, verbose=50)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    print('X:', featureX.shape)
    print('Y:', featureY.shape)
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    nrows = x_data.shape[0]
    print(nrows)
    array = x_data
    similar = np.zeros((nrows, nrows))
    bian = np.full((nrows, nrows), 0, dtype=int)
    n = 20
    d = np.full((nrows), n, dtype=int)
    print(array)
    for i in range(nrows):
        for j in range(nrows):
            if i == j:
                similar[i][j] = 0
            else:
                data0 = np.mat([array[i], array[j]])
                avg = np.mean(data0, 0)
                similar[i][j] = AdjustedCosine(data0[0, :], data0[1, :], avg)
    similar0 = np.array(similar)
    print(similar)
    for i in range(nrows):
        for j in range(nrows):
            if d[i] > 0:
                b = np.argmax(similar[i])
                if d[b] > 0:
                    bian[i][b] = 1
                    bian[b][i] = 1
                    d[b] = d[b] - 1
                    similar[i][b] = 0
                    d[i] = d[i] - 1
                else:
                    similar[i][b] = 0
                    continue

    for i in range(nrows):
        for j in range(nrows):
            sheet.cell(i + 1, j + 4).value = bian[i][j]

    wb.save("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\new_lncrna\\luan660\\2000\\newnode.xlsx")

    #构建图结构
    nodes = 600
    G = dgl.DGLGraph()
    G.add_nodes(nodes)
    edges_u = []
    edges_v = []
    for row, u in enumerate(bian):
        for column, v in enumerate(u):
            if int(float(v)):
                edges_u.append(row)
                edges_v.append(column)
    G.add_edges(edges_u, edges_v)

    # plt.figure(figsize=(20, 14))
    # color_map = []
    # for node in nx.nodes(G):
    #     if label[node] == 0.0:
    #         color_map.append('#e3b4b8')
    #     elif label[node] == 1.0:
    #         color_map.append('#8076a3')
    #     elif label[node] == 2.0:
    #         color_map.append('#93b5cf')
    #     elif label[node] == 3.0:
    #         color_map.append('#9abeaf')
    #     else:
    #         color_map.append('#e2e7bf')
    # nx.draw(G.to_networkx(), edge_color='#a7a8bd', node_color=color_map, node_size=120, with_labels=False)
    # plt.show()

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])
    return x_data,similar0,G



