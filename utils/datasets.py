import os
import torch.utils.data
from collections import Counter
import numpy as np
np.set_printoptions(precision=4, suppress=True)
os.environ["OMP_NUM_THREADS"] = "1"


class DataLoader(object):
    def __init__(self, data_path, batch_size=1, shuffle=True):
        self.datasets = BurnupDatasets(data_path)
        workers = np.min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        # print(workers)
        # exit()
        self.data_loader = torch.utils.data.DataLoader(
            self.datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True
        )


class BurnupDatasets(torch.utils.data.Dataset):
    def __init__(self, data_path):
        assert os.path.exists(data_path), f'Error. Datafile: {data_path} is not exists.'
        with open(data_path, 'r') as f:
            data = f.readlines()
        self.data = np.array([x.strip('\n').replace(' ', '').split(',') for x in data], dtype=object)
        self.items = ['rel_p', 'volume', 'cz', 'HN0', 'ftype', 'm92238', 'm92235', 'm90232', 'power', 'max_bp']
        self.methods = ['discard', 'one-hot', 'std', 'norm', 'one-hot', 'norm', 'norm', 'norm', 'one-hot', 'std']
        assert len(self.items) == len(self.methods)
        self.analyse_data()

    def analyse_data(self):
        data = self.data
        self.kinds_dict = {}
        for i, item_name in enumerate(self.items):
            # i: 1 item_name: volume
            # print(data,data.shape)
            item = data[:, i]  # 数据的所有行的第i列
            # print(item, item.shape)
            # exit()
            m = self.methods[i]  # i=0， m = discard
            if m == 'one-hot':
                count_dict = dict(Counter(item))
                print(count_dict)
                keys = sorted(list(count_dict.keys()))
                print(keys)
                self.kinds_dict[item_name] = keys
                # print(count_dict, keys)
            elif m == 'std':
                item_array = np.array(item, dtype=np.float32)
                self.kinds_dict[item_name] = [np.mean(item_array), np.std(item_array)]
            elif m == 'norm':
                item_array = np.array(item, dtype=np.float32)
                self.kinds_dict[item_name] = [np.min(item_array), np.max(item_array)]
            else:
                continue
        print(f'kinds_dict: {self.kinds_dict}')

    def __len__(self):
        return len(self.data)

    def encode(self, item_name, value, method):
        itn, v, m = item_name, value, method
        if m == 'discard':
            return []
        elif m == 'one-hot':
            kinds = self.kinds_dict[itn]
            coded = [0] * len(kinds)  # [0, 0, 0, 0, 0, 0, 0, 0, 0]
            coded[kinds.index(v)] = 1
            # print(kinds, v, coded)
            return coded
        elif m == 'norm':
            vmin, vmax = self.kinds_dict[itn]
            v_norm = (float(v) - vmin) / (vmax - vmin)
            return [v_norm]
        elif m == 'std':
            vmean, vstd = self.kinds_dict[itn]
            # print(v, vmean, vstd)
            v_std = (float(v) - vmean) / vstd
            return [v_std]

    def __getitem__(self, idx):
        sdata = self.data[idx]  # single data
        # print(sdata)
        # rel_p, volume, cz, HN0, ftype, m92238, m92235, m90232, power, max_bp = sdata
        assert len(sdata) == len(self.items), f'len(sdata) != len(self.items), items: {self.items}'
        arrow = []
        for i, item_name in enumerate(self.items):
            v = sdata[i]  # value
            m = self.methods[i]  # method
            tmp = self.encode(item_name=item_name, value=v, method=m)

            arrow.extend(tmp)
            # print(i, item_name, v, m, tmp)
        arrow = np.array(arrow, dtype=np.float32)
        target = np.array(arrow[-1], dtype=np.float32)
        # target = arrow[-1]
        arrow = arrow[:-1]
        # print(f'arrow: {arrow} {arrow.shape} target: {target:.6f}')
        # exit()
        return arrow, target


if __name__ == '__main__':
    bd = BurnupDatasets('../scripts/train.txt')

    for j, data in enumerate(bd):
        arrow, target = data
        print(j, arrow, arrow.shape, target)
        # exit()
