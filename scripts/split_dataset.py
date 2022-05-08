"""
split dataset into train.txt and test.txt
"""
import random
from copy import deepcopy
from pathlib import Path


def run(txt_path, num_test_sample=100):
    with open(txt_path, 'r') as f:
        data = f.readlines()
    nto = len(data)  # num_total
    nts = num_test_sample
    random.shuffle(data)
    train_data = sorted(deepcopy(data[:nto-nts:]))  # 前
    test_data = sorted(deepcopy(data[nto-nts::]))

    pt = Path(txt_path).parent
    with open(f'{pt}/train.txt', 'w') as f:
        f.writelines(train_data)
    with open(f'{pt}/test.txt', 'w') as f:
        f.writelines(test_data)


if __name__ == '__main__':
    run('./dataset.txt')
