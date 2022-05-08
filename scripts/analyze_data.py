import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class AnalyzeDataset(object):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            data = f.readlines()
        self.data = [x.strip('\n').replace(' ', '').split(',') for x in data]

        # print(len(data), data[0])
        self.items = ['rel_p', 'volume', 'cz', 'HN0', 'ftype', 'm92238', 'm92235', 'm90232', 'power', 'max_bp']

    def run(self):
        data = np.array(self.data, dtype=object)

        for i, item_name in enumerate(self.items):
            item = data[:, i]
            # volumes = data[:, 1]
            # print(np.unique(volumes))
            # a = Counter(volumes)
            # print(a)
            # print(np.unique(item))
            count_dict = dict(Counter(item))
            item_count = len(count_dict.keys())
            print(f'item_name: {item_name:<6} count: {item_count:<3} ', end='')
            if item_count < 100:
                print(f'count_dict: {count_dict}')
            else:
                print('')

            if item_name in ['cz', 'HN0', 'm92238', 'm92235', 'm90232', 'max_bp']:
                it = np.array(item, dtype=np.float32)
                print(it.shape, np.min(it), np.max(it), np.mean(it))


if __name__ == '__main__':
    ad = AnalyzeDataset(data_file='dataset.txt')
    ad.run()
