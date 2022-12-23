import argparse
import copy
import csv
import os
import pickle
import numpy as np
import openpyxl
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix, multilabel_confusion_matrix, classification_report
from similar import similar_str
from losses import LDAMLoss
from matrix import metric
from select_feat import feature_selection
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model1 import  GAT, CorrectAndSmooth


torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

beta = 0.999
samples_per_cls = [233, 119, 72, 34, 20]
no_of_classes = 5 # 几类
b11 = 1.5

with open('results.txt', 'a') as f:
    f.write(str(11111) + '\n')


def loadCSV(csvf):
    dictGraphsLabels = {}
    dictLabels = {}
    dictGraphs = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)
        for i, row in enumerate(csvreader):
            filename = row[1]
            g_idx = int(filename.split('_')[0])  # 图的序号
            label = row[2]
            # dictGraphs = {g_idx1 : [filename1, filename2, ...], g_idx : [filename1, filename2, ...], ...}
            if g_idx in dictGraphs.keys():
                dictGraphs[g_idx].append(filename)
            else:
                dictGraphs[g_idx] = [filename]
                dictGraphsLabels[g_idx] = {}
            if label in dictGraphsLabels[g_idx].keys():
                dictGraphsLabels[g_idx][label].append(filename)
            else:
                dictGraphsLabels[g_idx][label] = [filename]
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels, dictGraphs, dictGraphsLabels


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    # load data
    feats00 = np.load('E:\\duoladuola\\相关论文\\GM_att\\data\\600\\features6.npy', allow_pickle=True)
    print('feature dimension: ', feats00.squeeze(0).shape)
    feats00 = torch.tensor(feats00.squeeze(0)).type(torch.FloatTensor)
    with open('E:\\duoladuola\\相关论文\\GM_att\\data\\label600_0.pkl', 'rb') as f:
        label = pickle.load(f)

    label = list(label.values())
    label = [[i] for i in label]
    label = torch.tensor(np.array(label)).type(torch.LongTensor)

    max_acc = 0
    db_train = []
    db_val = []
    db_test = []

    for k_fold in range(5):
        dictLabels0, dictGraphs0, dictGraphsLabels0 = loadCSV(
            'E:\\duoladuola\\相关论文\\GM_att\\data\\600_5K\\train_' + str(k_fold) + '.csv')
        db_train.append(dictLabels0)
        dictLabels1, dictGraphs1, dictGraphsLabels1 = loadCSV(
            'E:\\duoladuola\\相关论文\\GM_att\\data\\600_5K\\test_' + str(k_fold) + '.csv')
        db_val.append(dictLabels1)
        dictLabels2, dictGraphs2, dictGraphsLabels2 = loadCSV(
            'E:\\duoladuola\\相关论文\\GM_att\\data\\600_5K\\test_' + str(k_fold) + '.csv')
        db_test.append(dictLabels2)


    n_classes = len(np.unique(np.array(list(label))))
    print('There are {} classes '.format(n_classes))

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0).squeeze()

    acc_total = []
    auc_total = []
    Recall_total = []
    Sn = []
    Sp = []
    Mcc = []
    F1 = []

    for k_fold in range(5):
        wb = openpyxl.Workbook()
        sheet = wb.active
        print('------ ', k_fold + 1, 'fold ------')
        train_idx = []
        valid_idx = []
        test_idx = []
        for i, k in enumerate([db_train[k_fold]]):  # k:图序号     v:{标签1: [该图中含该标签的节点], 标签2: [], ...}
            for m, n in k.items():
                train_idx.append([int(a.split('_')[1]) for a in n])

        for i, k in enumerate([db_val[k_fold]]):
            for m, n in k.items():
                valid_idx.append([int(a.split('_')[1]) for a in n])

        for i, k in enumerate([db_test[k_fold]]):
            for m, n in k.items():
                test_idx.append([int(a.split('_')[1]) for a in n])

        train_idx = torch.tensor([y for x in train_idx for y in x]).type(torch.LongTensor)
        test_idx = torch.tensor([y for x in test_idx for y in x]).type(torch.LongTensor)
        valid_idx = torch.tensor([y for x in valid_idx for y in x]).type(torch.LongTensor)
        args.pretrain = True
        # 先训练再做后处理
        if args.pretrain:
            # 加载数据
            feats = np.load("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats\\6mer_2000_94.8\\" + "feat" + str(k_fold)+".npy", allow_pickle=True)
            print('feature dimension: ', feats.shape)
            feats = torch.tensor(feats)
            with open("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats\\6mer_2000_94.8\\" + "G_" + str(k_fold) + ".pkl", 'rb') as f:
                dgl_graph = pickle.load(f)
            G0 = dgl_graph[0]

            n_features = len(feats[0])  # 特征维
            # load model
            if args.model == 'gat':
                model = GAT(in_feats=n_features,
                            n_classes=n_classes,
                            n_hidden=args.hid_dim,
                            n_layers=args.num_layers,
                            n_heads=args.n_heads,
                            activation=F.relu,
                            dropout=args.dropout,
                            attn_drop=args.attn_drop)
            else:
                raise NotImplementedError(f'Model {args.model} is not supported.')
            model = model.to(device)
            print('---------- Before ----------')
            model.load_state_dict(torch.load(f'base/{args.model}.pt'), strict=False)
            model.eval()
            y_soft = model(G0, feats).exp()
            y_pred = y_soft.argmax(dim=-1, keepdim=True)

            valid_acc = y_pred[valid_idx].eq(label[valid_idx]).float().mean()
            test_acc = y_pred[test_idx].eq(label[test_idx]).float().mean()
            print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')
            a = y_pred.detach().numpy().squeeze(1)
            h = 5
            G = similar_str(a, h, k_fold)
            print(G)
            print('---------- Correct & Smoothing ----------')
            cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                                  correction_alpha=args.correction_alpha,
                                  correction_adj=args.correction_adj,
                                  num_smoothing_layers=args.num_smoothing_layers,
                                  smoothing_alpha=args.smoothing_alpha,
                                  smoothing_adj=args.smoothing_adj,
                                  scale=args.scale)

            mask_idx = torch.cat([train_idx])
            y_soft = cs.correct(G, y_soft, label[mask_idx], mask_idx)
            y_soft = cs.smooth(G, y_soft, label[mask_idx], mask_idx)
            y_pred = y_soft.argmax(dim=-1, keepdim=True)
            valid_acc = y_pred[valid_idx].eq(label[valid_idx]).float().mean()
            test_acc = y_pred[test_idx].eq(label[test_idx]).float().mean()

            print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')

            y_pred0 = y_pred[test_idx].squeeze(1)
            y_true0 = label[test_idx].squeeze(1)
            confu_matrix = confusion_matrix(y_true0, y_pred0)

            sn, sp, mcc, f1 = metric(confu_matrix)
            Sn.append(sn)
            Sp.append(sp)
            Mcc.append(mcc)
            Recall = np.array(sn).mean()
            Recall_total.append(Recall)
            f1 = np.array(f1).mean()
            F1.append(f1)
            print('Test Recall:', Recall)
            print('Test F1:', f1)
            print("Sn: ", sn, "\n", "Sp: ", sp, "\n", "Mcc:", mcc)

            y_pred = F.one_hot(y_pred[test_idx], num_classes=5).squeeze(1)
            y_label = F.one_hot(label[test_idx], num_classes=5).squeeze(1)
            auc_score = roc_auc_score(y_label, y_pred)
            print("AUC of Test Samples: ", auc_score)
            acc_total.append(test_acc)
            auc_total.append(auc_score)
            model.zero_grad()

        else:
            # 特征选择，构图
            feats, simi, G0 = feature_selection(feats00, label, train_idx, 2000)
            for i in range(600):
                for j in range(600):
                    sheet.cell(i + 1, j + 1).value = simi[i][j]
            wb.save("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats\\6mer_2000_94.8\\" + "feat_simi_" + str(k_fold) + ".xlsx")
            feats = torch.tensor(feats).type(torch.FloatTensor)
            np.save("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats\\6mer_2000_94.8\\" + "feat" + str(k_fold), arr=feats)
            with open("E:\\duoladuola\\相关论文\\GM_att\\data\\600\\select_feats\\6mer_2000_94.8\\" + "G_" + str(k_fold) + ".pkl",'wb') as f:
                pickle.dump([G0], f)
            n_features = len(feats[0])
            # load model
            model = GAT(in_feats=n_features,
                            n_classes=n_classes,
                            n_hidden=args.hid_dim,
                            n_layers=args.num_layers,
                            n_heads=args.n_heads,
                            activation=F.relu,
                            dropout=args.dropout,
                            attn_drop=args.attn_drop)
            model = model.to(device)
            print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
            opt = optim.RMSprop(model.parameters(), lr=args.lr)


            best_acc = 0
            model.zero_grad()
            best_model = copy.deepcopy(model)

            def adjust_learning_rate(optimizer, lr, epoch):
                if epoch <= 50:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr * epoch / 50

            # training
            print('---------- Training ----------')
            for i in range(args.epochs):
                adjust_learning_rate(opt, args.lr, i)
                model.train()
                opt.zero_grad()
                logits = model(G0, feats)

                criterion = LDAMLoss(cls_num_list=samples_per_cls, max_m=0.7, s=30, weight=weights)
                train_loss = criterion(logits[train_idx], label.squeeze(1)[train_idx])
                train_loss = (train_loss - b11).abs() + b11
                train_loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    logits = model(G0, feats)
                    y_pred = logits.argmax(dim=-1, keepdim=True)
                    train_acc = y_pred[train_idx].eq(label[train_idx]).float().mean()
                    valid_acc = y_pred[valid_idx].eq(label[valid_idx]).float().mean()
                    print(
                        f'Epoch {i} | Train loss: {train_loss.item():.4f} | Train acc: {train_acc:.4f} | Valid acc {valid_acc:.4f}')
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        best_model = copy.deepcopy(model)
            # testing & saving model
            print('---------- Testing ----------')
            best_model.eval()


            logits = best_model(G0, feats)
            y_pred = logits.argmax(dim=-1, keepdim=True)
            test_acc = y_pred[test_idx].eq(label[test_idx]).float().mean()
            print(f'Test acc: {test_acc:.4f}')
            if not os.path.exists('base'):
                os.makedirs('base')
            torch.save(best_model.state_dict(), f'base/{args.model}.pt')
            torch.save(best_model,  f'base/model0.pt')
            best_model.zero_grad()

            y_pred1 = F.one_hot(y_pred[test_idx], num_classes=5).squeeze(1)
            y_label1 = F.one_hot(label[test_idx], num_classes=5).squeeze(1)
            auc_score = roc_auc_score(y_label1, y_pred1)
            print("AUC of Test Samples: ", auc_score)
            acc_total.append(test_acc)
            auc_total.append(auc_score)

            y_pred0 = y_pred[test_idx].squeeze(1)
            y_true0 = label[test_idx].squeeze(1)
            confu_matrix = confusion_matrix(y_true0, y_pred0)
            sn, sp, mcc, f1 = metric(confu_matrix)
            Sn.append(sn)
            Sp.append(sp)
            Mcc.append(mcc)
            Recall = np.array(sn).mean()
            Recall_total.append(Recall)
            f1 = np.array(f1).mean()
            F1.append(f1)
            print('Test Recall:', Recall)
            print('Test F1:', f1)
            print("Sn: ", sn, "\n", "Sp: ", sp, "\n", "Mcc:", mcc)

    print('---------- Total ----------')
    print('Total Acc:', str(np.array(acc_total).mean())[:5])
    print('Total Auc:', str(np.array(auc_total).mean())[:5])
    print('Total F1:', str(np.array(F1).mean())[:5])
    print('Total Recall:', str(np.array(Recall_total).mean())[:5])
    Sn = np.array(Sn)
    Sn = np.mat(Sn)
    Sn = np.mean(Sn, 0)
    print('Total Sn:{}'.format(Sn))
    Sp = np.array(Sp)
    Sp = np.mat(Sp)
    Sp = np.mean(Sp, 0)
    print('Total Sp:{}'.format(Sp))
    Mcc = np.array(Mcc)
    Mcc = np.mat(Mcc)
    Mcc = np.mean(Mcc, 0)
    print('Total Mcc:{}'.format(Mcc))

    with open('results.txt', 'a') as f:
        f.write('------ total fold ------\n')
        f.write('Acc:' + str(np.array(acc_total).mean())[:5] + '\n')
        f.write('Auc:' + str(np.array(auc_total).mean())[:5] + '\n')
        f.write('F1:' + str(np.array(F1).mean())[:5] + '\n')
        f.write('Recall:' + str(np.array(Recall_total).mean())[:5] + '\n')
        f.write('Sn:' + str(Sn) + '\n')
        f.write('Sp:' + str(Sp) + '\n')
        f.write('Mcc:' + str(Mcc) + '\n')


if __name__ == '__main__':
    """
    Hyperparameters
    """
    parser = argparse.ArgumentParser(description='GATLncLoc+C&S')

    # Dataset
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    # Base predictor
    parser.add_argument('--model', type=str, default='gat')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hid-dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=500)
    # extra options for gat
    parser.add_argument('--n-heads', type=int, default=3)
    parser.add_argument('--attn_drop', type=float, default=0.003)
    # C & S
    parser.add_argument('--pretrain', action='store_true', help='Whether to perform C & S')
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0.9)
    parser.add_argument('--correction-adj', type=str, default='DAD')
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0.2)
    parser.add_argument('--smoothing-adj', type=str, default='DAD')
    parser.add_argument('--scale', type=float, default=5.)

    args = parser.parse_args()
    print(args)

    main()