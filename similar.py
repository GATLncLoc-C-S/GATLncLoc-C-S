import csv
import pickle

import networkx as nx
import matplotlib.pyplot as plt
import Levenshtein
import openpyxl
import numpy as np
import dgl

def similar_str(a,h, k_fold):
    num_node = 600
    array = np.zeros((num_node,num_node))
    nrows = 600

    bian = np.full((nrows, nrows), 0, dtype=int)
    array0 = np.zeros((num_node, num_node))
    d = np.full((nrows), h, dtype=int)
    # print(d)
    for i in range(num_node):
        for j in range(num_node):
            if i==j:
                array[i][j] = 0
            else:
                array[i][j] = Levenshtein.ratio(str(a[i]), str(a[j]))

    for i in range(nrows):
        for j in range(nrows):
            if d[i]>0:
                b = np.argmax(array[i])
                if d[b]>0:
                    bian[i][b] = 1
                    bian[b][i] = 1
                    d[b]=d[b]-1
                    array[i][b] = 0
                    d[i]=d[i]-1
                else:
                    array[i][b]=0
                    continue

    nodes = num_node
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

    return G


