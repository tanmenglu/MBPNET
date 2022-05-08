"""读取原始数据，产生数据集"""
import os
import numpy as np
from pathlib import Path
from dm import DmFile
from copy import deepcopy
import pandas as pd


class GenerateDatasets(object):
    def __init__(self):
        self.vp_dict = {'100': 50, '125': 100, 'inf': 1, '150': 170, '200': 400, '250': 780, '300': 1350, '500': 6250}  # volume-power dict
        self.HN_thresh = 0.12

    def generate(self):
        save_file = 'dataset.txt'
        if os.path.exists(save_file):
            os.system(f'rm {save_file}')

        path = "/Volumes/Untitled/tmll/tanmenglu/BoShiKeTi"
        # flag = 0
        for i, (root, dirs, files) in enumerate(os.walk(path)):
            if root[-3::] != 'OUT':
                continue
            if not (root.split('/')[6] in ['enrich', 'TH', 'U']):
                continue
            if root.split('/')[-2][:2:] == 're':
                continue
            # print(i, root)
            rel_p, volume, cz, HN0, ftype, m92238, m92235, m90232, power, max_bp = self.read(path=root, flag=0)
            content = f'{rel_p:<30},{volume:<3},{cz:>5f},{HN0:<5.2f},{ftype:<2},{m92238:.5f},{m92235:.5f},{m90232:.5f},{power:<4},{max_bp:.4f}'
            print(content)
            os.system(f'echo "{content}" >>{save_file}')

    def read(self, path, flag):
        # if flag == 1:
        #     path = '/Volumes/Untitled/tmll/tanmenglu/BoShiKeTi/TH/burnup/10/05/02/n100502OUT'
        rel_p = '/'.join(deepcopy(path).split('/')[6::])
        calculate_type = rel_p.split('/')[0]  # enrich, Th, U
        # print(rel_p)
        # heavymass
        file = f'{path}/HEAVYMASS'
        # print(file)

        assert os.path.exists(file), f'file: {file} is not exists.'
        with open(file, 'r') as f:
            heavymass = f.readlines()[1::]
        heavymass = [x.strip('\n').split() for x in heavymass if x != '\n']  # [n, 3]
        heavymass = np.array(heavymass, dtype=np.float32)

        # origin
        file = f'{path}/{Path(path).stem.strip("OUT")}-1-1'
        assert os.path.exists(file)
        dmfile = DmFile(file)
        # 读取体积
        try:
            line = dmfile.gt(k='*2', p=1, sep=None, im='sa', k2=None)  # 找得到
            volume = 'inf'
        except:
            # print(dmfile.parts[1])
            volume = float(dmfile.parts[1][1].split()[-1])
            volume = str(int(volume))
        # 读取燃料通道半径cz
        cz = float(dmfile.gt(k='20', p=1, sep=None, im='sa').split()[-1])
        # 读取燃料成分HN和类型ftype

        m92238 = float(dmfile.gt(k='92238.90c', p=2, im='s0').split()[-1])
        m92235 = float(dmfile.gt(k='92235.90c', p=2, im='s0').split()[-1])
        try:
            m90232 = float(dmfile.gt(k='90232.90c', p=2, im='s0').split()[-1])
            ftype = 'Th'  # fuel type
        except:
            m90232 = 0.
            ftype = 'U'

        # 计算HN
        HN0 = m92238 + m92235 + m90232
        MATn = heavymass[..., 2]
        MAT1 = heavymass[0, 2]
        N = MATn / MAT1 * HN0
        HN = N/(N+88)
        # print('mat1', MAT1)
        # print('x', HN, HN.shape)
        # 计算功率
        power = self.vp_dict[volume]
        # 计算转化率
        CN = (m92235/(m92235+m92238)*100-0.25)/(0.714-0.25)
        # print(f'CN {CN} {self.CN20}')
        # 计算burnup
        enrich = 20 if calculate_type in ['TH', 'U'] else int(rel_p.split('/')[1])
        # print(f'enrich: {enrich}')
        CN_enrich = (enrich - 0.25)/(0.714 - 0.25)
        Burnup = (1/(MAT1*CN + (MATn-MAT1)*CN_enrich))*power*heavymass[..., 0]
        # print(f'Burnup {Burnup} {Burnup.shape}')
        # print(f'MAT1: {MAT1} {CN} {MAT1*CN} mn: {(MATn-MAT1)*self.CN20} ')
        valid_burnup = Burnup[HN <= self.HN_thresh]
        max_bp = np.max(valid_burnup)
        # print(f'vb: {valid_burnup} {valid_burnup.shape} maxbp: {max_bp}')
        # print(
        #     f'{Path(path).stem} volume: {volume} cz: {cz} 90323: {m90232} HN0: {HN0} power: {power}'
        #     f'')

        return rel_p, volume, cz, HN0, ftype, m92238, m92235, m90232, power, max_bp


if __name__ == '__main__':
    gd = GenerateDatasets()
    gd.generate()
