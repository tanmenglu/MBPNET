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
        flag = 0
        for i, (root, dirs, files) in enumerate(os.walk(path)):
            if root[-3::] != 'OUT':
                continue
            if not (root.split('/')[6] in ['enrich', 'TH', 'U']):
                continue
            if root.split('/')[-2][:2:] == 're':
                continue
            # a = os.listdir(root)
            # print(a, len(a))
            # b = []
            # for j in files:
                # if j[:4:] == 'MAT-':
                    # b.append(j)
            keff_data = self.read_keff_file(root)  # (n, 3), n: number of minor steps
            # print(keff_data, keff_data.shape)
            for j, (step, time, keff) in enumerate(keff_data):
                if step == '0-0':
                    continue
                mjrs, mnrs = step.split('-')  # major step, minor step
                ncld1, ncld2, ncld3 = self.read_mat_file(root, mjrs, mnrs)  # nuclide 1 2 3
                # TODO: 不同的MAT文件的不趋同区的元素种类会有差别，加载数据集时需要前处理
                rrn = self.read_o_file(root, mjrs, mnrs)  # reaction rate of nuclide
                print(ncld1.shape, ncld2.shape, ncld3.shape, rrn.shape)
                # print(j, root, step, time, keff, o_file)
                continue


            # print(b, len(b))
            # c = [x for x in files if x[:4:] == 'MAT-']
            # print(c, len(c))
            # d = [x for x in files if (x[:8:] == root[-10:-3]+'o' and x[-3::] != 'del')]
            # print(d, len(d))


            # break

            # print(i, root)
            # rel_p, volume, cz, HN0, ftype, m92238, m92235, m90232, power, max_bp = self.read(path=root, flag=flag)
            # content = f'{rel_p:<30},{volume:<3},{cz:>5f},{HN0:<5.2f},{ftype:<2},{m92238:.5f},{m92235:.5f},{m90232:.5f},{power:<4},{max_bp:.4f}'
            # print(content)
            # os.system(f'echo "{content}" >>{save_file}')

    def read_o_file(self, root, mjrs, mnrs):
        o_file = f'{root}/{root.split("/")[-1][:-3:]}o-{int(mjrs) + 1}-{int(mnrs) + 1}'
        if not os.path.exists(o_file):
            o_file = f'{root}/{root.split("/")[-1][:-3:]}o-{int(mjrs) + 1}-rel'
        assert os.path.exists(o_file), f'o file {o_file} does not exist.'
        with open(o_file, 'r') as f:
            data = f.readlines()
        # 获取反应率的编号
        idxx = [idx for idx, x in enumerate(data) if ('fm14' in x or 'sd14' in x)]  # 查找包围1000-1133的4种反应率的索引
        assert len(idxx) == 2
        chunk = data[idxx[0]+1:idxx[1]]
        irr = np.array([x.replace('(', '').replace(')', '').split()[2::] for x in chunk])  # identifier of reaction rate
        assert irr.shape[1] == 5
        irr = np.array([[f'{x[0]}.{x[1]}', f'{x[0]}.{x[2]}', f'{x[0]}.{x[3]}', f'{x[0]}.{x[4]}'] for x in irr])
        irr = irr.reshape(-1)
        irr = np.concatenate((np.array(['9999']), irr))
        # print(irr, irr.shape)

        o_data = np.zeros((len(irr), 2), dtype=object) - 2  # reaction rate of nuclide
        o_data[:, 0] = irr
        # print(o_data, o_data.shape)
        # 获取反应率的数值
        chunk = [data[idx]+data[idx+1] for idx, x in enumerate(data) if 'multiplier bin:' in x]
        for c in chunk:
            l1 = c.split('\n')[0]
            nucl, nir = l1.split()[-2::]  # nuclide, identifier of reaction rate of nuclide
            nucl, nir = ('9999', None) if nucl == 'bin:' else (nucl, nir)
            nirr = f'{nucl}.{nir}' if nir is not None else f'{nucl}'

            l2 = c.split('\n')[1]
            v = l2.split()[-2]  # value of nir

            row_idx = np.argwhere(o_data[:, 0] == nirr)
            o_data[row_idx, 1] = v
            # print(nucl, nir, nirr, v)
            # print(o_data)
        verify_empty = o_data[o_data[:, 1] == -2]
        assert len(verify_empty) == 0
        # print(o_data, o_data.shape)
        # print(verify_empty)
        # print(chunk)
        # exit()
        return o_data



    def read_mat_file(self, root, mjrs, mnrs):
        mat_file = f'{root}/MAT-{int(mjrs) + 1}-{mnrs}-0'
        # print(mat_file)
        assert os.path.exists(mat_file), f'MAT file {mat_file} does not exist.'
        with open(mat_file, 'r') as f:
            data = f.readlines()
        data = np.array([x.split() for x in data[:-1:]], dtype=object)
        nuclide = []
        for area in ['1', '2', '3']:
            tmp = data[data[:, 0] == area][:, 1::].reshape((-1, 2))  # (n, 2)
            tmp = tmp[tmp[:, 0] != '0']
            # tmp = np.array(tmp, dtype=np.float32)
            # print(area, tmp, tmp.shape)
            nuclide.append(tmp)
        return nuclide


    def read_keff_file(self, root):
        assert os.path.exists(f'{root}/KEFF'), f'KEFF is not exists.'
        with open(f'{root}/KEFF', 'r') as f:
            data = f.readlines()
        # print(data, len(data))
        keff_data = [x.split()[:3] for x in data]
        keff_data = np.array(keff_data, dtype=object)
        # print(keff_data, keff_data.shape)
        return keff_data

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
