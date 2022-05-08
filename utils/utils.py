import os
import numpy as np
import torch


def select_device(device='', batch_size=None, is_K80=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        if is_K80:
            # zzd k80
            ng = 2
        else:
            ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # print(output.shape)
    # print(maxk, batch_size)

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    # print(target)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)

    res = torch.Tensor((len(topk)))
    res[:] = 0  # init
    for i, k in enumerate(topk):
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res[i] = correct_k.mul_(100.0 / batch_size)
    return res


def cal_mae_mse_rmse(pd, gt):
    """
    calculate errors.
    MAE: Mean Absolute Error.
    MSE: Mean Square Error.
    RMSE: Root Mean Square Error.
    :param pd: predict value
    :param gt: ground truth
    :return: errors. tuple (3,)
    """
    assert len(pd) == len(gt)
    mae = np.sum(np.abs(pd-gt))/len(pd)
    mse = np.sum(np.square(pd-gt))/len(pd)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

