import argparse
import torch
import torch.nn
import torch.optim
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from utils.datasets import DataLoader, BurnupDatasets
from models.BurnupNet import BurnupModel
from models.burnupsvm import SVMBurnupModel
from utils import utils


def train(args):
    learning_rate = 0.0001
    momentum = 0.9

    trt = args.train_txt
    tet = args.test_txt
    bs = args.batch_size
    epochs = args.epochs

    deivce = utils.select_device(device=args.device, batch_size=bs)
    train_loader = DataLoader(data_path=trt, batch_size=bs, shuffle=True)
    bp_mean, bp_std = train_loader.datasets.kinds_dict['max_bp']
    test_loader = DataLoader(data_path=tet, batch_size=bs, shuffle=False)
    if args.model == 'ANN':
        model = BurnupModel(num_ipt=23)
    elif args.model == 'SVM':
        model = SVMBurnupModel(num_ipt=23)
    else:
        raise NotImplementedError(f'model: {args.model} not implemented')

    model.to(deivce)

    if args.loss == 'MSELoss':
        lossfunction = torch.nn.MSELoss()  # 用来计算损失的对象
    elif args.loss == 'BCELoss':
        lossfunction = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'SmoothL1Loss':
        lossfunction = torch.nn.SmoothL1Loss()
    else:
        raise NotImplementedError(f'loss function {args.loss} not implemented.')
        # lossfunction = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_result = []
    test_result = []

    for epoch in range(epochs):
        model.train()
        outputs_train = np.zeros((0, 2), dtype=np.float32)
        for idx, data in enumerate(train_loader.data_loader):
            arrows, targets = data  # tensor (bs, 23)  tensor (bs, 1)

            preds = model(arrows)  # (bs, 23) --> (bs, 1)
            loss = lossfunction(preds, targets)

            print(f'\r[{epoch+1:>3}/{epochs}] [{idx:>2}] loss: {loss:.6f} ', end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outt = torch.cat((targets.view(-1, 1), preds.view(-1, 1)), dim=1)
            outt = outt.detach().cpu().numpy()  # [bs, 2]
            outputs_train = np.concatenate((outputs_train, outt), axis=0)  # (514, 2)
        scheduler.step()

        outputs_train = outputs_train * bp_std + bp_mean
        mae_train, mse_train, rmse_train = cal_mae(outputs_train)
        train_result.append([mae_train, mse_train, rmse_train])
        # print(f'out_train: {outputs_train.shape}, mae: {mae_train}')

        # test
        model.eval()
        outputs_test = np.zeros((0, 2), dtype=np.float32)
        for idx, data in enumerate(test_loader.data_loader):
            arrows, targets = data
            preds = model(arrows)
            out = torch.cat((targets.view(-1, 1), preds.view(-1, 1)), dim=1)
            out = out.detach().cpu().numpy()  # [bs, 2]
            outputs_test = np.concatenate((outputs_test, out), axis=0)
        outputs_test = outputs_test*bp_std + bp_mean
        mae_test, mse_test, rmse_test = cal_mae(outputs_test)
        test_result.append([mae_test, mse_test, rmse_test])

        print(f'mae_train: {mae_train:.4f} mae: {mae_test:.4f} mse: {mse_test:.4f} rmse: {rmse_test:.4f}')

    train_result = np.array(train_result)  # [epochs, 3]  每一代训练后训练集的mae mse和rmse
    test_result = np.array(test_result)  # [epochs, 3]

    test_mae = np.min(test_result, axis=0)[0]
    print(f'train: {np.min(train_result, axis=0)} test: {np.min(test_result, axis=0)}')
    plot(x=range(epochs), train_result=train_result, test_result=test_result,
         title=f'{args.model} {args.loss} {test_mae:.4f}')
    # 显示最终结果


def plot(x, train_result, test_result, title='xx'):
    marks = ['^', 's', 'p']
    plot_lines = ['mae', 'mse', 'rmse']
    plot_lines = plot_lines[:1:]
    for i, crit in enumerate(plot_lines):
        plt.plot(x, train_result[:, i], f'{marks[i]}-', color='#EE3B3B',
                 markersize=12, linewidth=3, label=f'{crit} train')
        plt.plot(x, test_result[:, i], f'{marks[i]}-', color='black',
                 markersize=12, linewidth=3, label=f'{crit} test')

    fontdict = {'size': 16, 'weight': 'bold'}
    plt.title(title, fontdict=fontdict)
    plt.xlabel(f'epochs', fontdict=fontdict)
    plt.ylabel(f'mae', fontdict=fontdict)
    # plt.xlim(-1,200)
    # plt.ylim(-1,5)
    plt.yticks(size=16, weight='bold')
    plt.xticks(size=16, weight='bold')

    plt.legend(prop=fontdict)
    plt.show()


def cal_mae(out):
    mae, mse, rmse = utils.cal_mae_mse_rmse(pd=out[:, 1], gt=out[:, 0])
    return mae, mse, rmse


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--train_txt', default='./scripts/train.txt', type=str, help='path to train file')
    paser.add_argument('--test_txt', default='./scripts/test.txt', type=str, help='path to test file')
    paser.add_argument('--epochs', default=50, type=int, help='epochs')
    paser.add_argument('--device', default='cpu', type=str, help='device for training')
    paser.add_argument('--batch-size', default=32, type=int, help='batch size')
    paser.add_argument('--model', default='SVM', type=str, help='choose model')
    paser.add_argument('--loss', default='MSELoss', type=str, help='choose loss function')
    args = paser.parse_args()

    models = ['ANN', 'SVM']
    lossf = ['BCELoss', 'MSELoss',  'SmoothL1Loss']
    args.model = models[0]
    args.loss = lossf[2]

    print(args.train_txt, args.epochs)
    # print(args.datafile)
    train(args)
