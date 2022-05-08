import os


class DmFile(object):
    def __init__(self, file):
        with open(file, 'r') as f:
            self.data = f.readlines()   # [line1, line2, ...]
            # print(f'file "{file}" is read.')
        self.parts = self.update_parts()  # [[part1], [[part2], ...]

    def update_parts(self):
        empty_idx = [i for i, x in enumerate(self.data) if x == '\n']  # 空行索引, [14, 45, 81]
        # print(empty_idx)
        assert len(empty_idx)+1 == 4 or len(empty_idx)+1 == 3  # assert 声明 必须是
        empty_idx = [0] + empty_idx + [len(self.data)]  # [14, 45, 81] --> [0, 14, 45, 81, 96]
        self.parts = [None] * (len(empty_idx) - 1)  # init [None, None, None, None]
        for i in range(len(empty_idx)-1):
            start_idx, end_idx = empty_idx[i:i+2]
            # self.parts[i] = [x for x in self.data[start_idx:end_idx] if x != '\n']
            self.parts[i] = self.data[start_idx:end_idx]
        return self.parts

    def update_data(self):
        data = []
        for x in self.parts:
            data = data + x
        self.data = data

    def m(self, k, p, c, rm, im=None, sp=None, k2=None, rk=None):
        self.modify(
            keyword=k, content=c, part=p, idx_mode=im, replace_mode=rm, separator=sp, keyword2=k2, replace_key=rk)

    def modify(
            self, keyword, content, part, replace_mode, idx_mode=None, separator=None, keyword2=None, replace_key=None):
        """
        定位，修改。
        数据范围：parts
        定位方式：直接给出索引，整行关键词索引all，split后整行索引sa，split后关键词索引si
        替换方式：整行替换all，split后第i个替换si
        还要考虑输入的替换为list或者定位的idx为list的情形，统一全部替换
        必选参数：
        :param keyword: 关键词
        :param part: 第x部分，输入值为索引。
        :param content: 修改内容
        :param replace_mode:
                'all': 整行替换，替换定位的idx行
                'si'：定位的idx行split后替换第i个元素
                'srk'：split之后根据replace_key进行替换
        可选参数：
        :param idx_mode:
                int i: i是数字，直接给出定位索引
                list: 直接给出多个idx组成的索引
                'all': 整行关键词索引
                'si': split后第i个为关键词。如's1':第1个.
                'sa': split后整行索引, 默认
        :param separator: split时的分隔符，None为连续空格分隔
        :param keyword2: 关键词2，注意：idx_mode为si时不可用。
        :return:
        """
        idx_mode = idx_mode if idx_mode is not None else 'sa'

        data = self.data if part == 'all' else self.parts[part]
        keywords = [keyword] if keyword2 is None else [keyword, keyword2]
        # 定位
        idx = self._get_index(idx_mode=idx_mode, data=data, keywords=keywords, separator=separator)
        print(f'{"":=<40}\nstart modify indexes: {idx}')
        # print([part[i] for i in idx])

        # 替换
        if replace_mode == 'all':
            data = self._replace(data, idx, content, part)
        elif replace_mode.startswith('s') and replace_mode[1::].isdigit():
            digit = int(replace_mode[1::])
            assert type(idx) == list
            for id in idx:
                line = data[id].split(separator)  # line string to list
                new_line = self._get_new_line_by_replace_content(
                    digit=digit, line=line, content=content, part=part, data=data, id=id)
                data = self._replace(data, id, new_line, part)
        elif replace_mode == 'srk':
            rk = replace_key
            assert rk is not None
            assert type(idx) == list
            for id in idx:
                line = data[id].split(separator)  # line string to list
                digit = self._get_srk_replace_index(line=line, rk=rk)
                new_line = self._get_new_line_by_replace_content(
                    digit=digit, line=line, content=content, part=part, data=data, id=id)
                data = self._replace(data, id, new_line, part)
        else:
            raise NameError(f'Undefied replace_mode for dashamei: {replace_mode}')

        self._update_data_or_parts(part, data)  # 更新数据

    def replace_90c_to_10c(self):
        """
        专有函数，把m1行到m2行中间所有行的.90c替换成.10c
        :return:
        """
        data, part = self.data, 'all'
        # 搜索索引
        m1_idx = [i for i, x in enumerate(data) if 'm1' in x][0]
        m2_idx = [i for i, x in enumerate(data) if 'm2' in x][0]
        for idx in range(m1_idx, m2_idx):
            data[idx] = data[idx].replace('.90c', '.10c')
        # data = data[:m1_idx] + [data[i].replace('.90c', '.10c') for i in range(m1_idx, m2_idx)] + data[m2_idx::]
        self._update_data_or_parts(part, data)  # 更新数据

    def d(self, k, p, rmm='current', icld=True, im=None, sep=None, k2=None):
        self.delete(keyword=k, part=p, remove_mode=rmm, include=icld, idx_mode=im, separator=sep, keyword2=k2)

    def delete(self, keyword, part, remove_mode='current', include=True, idx_mode=None, separator=None, keyword2=None):
        """
        删除内容。
        :param part:
                'all': 全部
                i: int, 0 1 2，第i部分
        :param remove_mode:
                "current": current line
                "after": all after lines
        :param include: remove_mode为after时生效，删除时是否包含当前行
        :return:
        """
        idx_mode = idx_mode if idx_mode is not None else 'sa'
        data = self.data if part == 'all' else self.parts[part]
        keywords = [keyword] if keyword2 is None else [keyword, keyword2]
        # print(idx_mode, len(data), keywords, separator)
        # exit()
        idx = self._get_index(idx_mode=idx_mode, data=data, keywords=keywords, separator=separator)
        # print(f'{"":=<40}\nstart modify indexes: {idx}')

        if remove_mode == 'current':
            for id in idx:
                data.pop(id)
        elif remove_mode == 'after':
            assert len(idx) == 1
            if include:
                del(data[idx[0]:])  # delete
            else:
                del(data[idx[0]+1:])
        else:
            raise NameError(f'Undefied remove_mode for dashamei: {remove_mode}')

        self._update_data_or_parts(part, data)  # 更新数据

        # 锤儿妹你在干啥

    def insert(self, content, idx=0, part=None):
        """
        注意：是在idx之前插入值
        :param content:
        :param idx:
        :param part:
        :return:
        """
        part = part if part is not None else 'all'
        data = self.data if part == 'all' else self.parts[part]
        content = content if content.endswith('\n') else f'{content}\n'
        data.insert(idx, content)
        self._update_data_or_parts(part, data)

    def append(self, content, part=None):

        part = part if part is not None else 'all'
        data = self.data if part == 'all' else self.parts[part]
        content = content if content.endswith('\n') else f'{content}\n'
        data.append(content)
        self._update_data_or_parts(part, data)

    def gt(self, k, p, im=None, sep=None, k2=None):
        return self.get_line(keyword=k, part=p, idx_mode=im, separator=sep, keyword2=k2)

    def get_line(self, keyword, part, idx_mode=None, separator=None, keyword2=None):
        idx_mode = idx_mode if idx_mode is not None else 'sa'
        data = self.data if part == 'all' else self.parts[part]
        keywords = [keyword] if keyword2 is None else [keyword, keyword2]
        idx = self._get_index(idx_mode=idx_mode, data=data, keywords=keywords, separator=separator)
        # print(keyword, part, idx_mode, separator, keyword2, idx)
        # assert len(idx) == 1  # 搜索到的索引不为1
        if len(idx) != 1:
            # print(f'error: idx: {idx}, len(idx) != 1')
            assert len(idx) == 1
        return data[idx[0]]

    def _get_srk_replace_index(self, line, rk):
        for i, l in enumerate(line):
            # print(line, i, l)
            if rk in l:
                return i
        raise NameError(f'replace key: "{rk}" not found in line: "{line}"')

    def _get_new_line_by_replace_content(self, digit, line, content, part, data, id):
        if digit >= len(line):
            raise NameError(f'given replace index "{digit}" is out of len(line), part: {part} idx: {id} line: {line}, '
                            f'please check input arguments.')
        line[digit] = content  # modify list element
        new_line, line_idx = '', 0
        # print([s for s in data[id]])
        for i, s in enumerate(data[id][:-1]):  # recover line to origin string form
            if s == ' ':
                new_line += s
            elif s != ' ' and (data[id][i + 1] == ' ' or data[id][i + 1] == '\n'):  # 当前非空格且后面一维是空格，则把值写进去。
                new_line += line[line_idx]
                line_idx += 1
            else:
                pass
        return new_line

    def _get_index(self, idx_mode, data, keywords, separator):
        if type(idx_mode) == int:
            idx = [int(idx_mode)]
        elif type(idx_mode) == list:
            idx = idx_mode
        elif idx_mode == 'all':
            # idx = [i for i, x in enumerate(part) if keyword in x]
            idx = [i for i, x in enumerate(data) if self._keywords_in_x(keywords, x, x_type='str')]
        elif idx_mode == 'sa':
            idx = [i for i, x in enumerate(data) if self._keywords_in_x(keywords, x.split(separator), x_type='list')]
        elif idx_mode.startswith('s') and idx_mode[1::].isdigit():
            digit = int(idx_mode[1::])
            # print(digit)
            idx = [i for i, x in enumerate(data) if self._keyword_in_x_si(keywords[0], x, separator, digit)]
        else:
            raise NameError(f'Undefied idx_mode for dashamei: {idx_mode}')
        assert type(idx) == list
        # print(idx, len(data))
        # exit()
        for x in idx:
            assert x < len(data)
        if len(keywords) > 1 and len(idx) > 1:  # keyword2 is not None
            raise NameError(
                f'ERROR: Idx are more than one while "keyword2" is not None. '
                f'keywords: "{keywords}" idx: {idx}')
        return idx

    def _update_data_or_parts(self, part, data):
        if part == 'all':
            self.data = data
            self.update_parts()
        else:
            self.parts[part] = data
            self.update_data()

    def _keyword_in_x_si(self, keyword, x, sep, digit):
        if digit >= len(x.split(sep)):
            return False
        else:
            ret = keyword in x.split(sep)[digit]
            return ret

    def _keywords_in_x(self, keywords, x, x_type='str'):
        # 默认逻辑为and
        assert type(keywords) == list
        assert type(x) == eval(x_type)
        for keywd in keywords:
            if keywd not in x:
                return False
        return True

    def _replace(self, data, ridx, content, part):
        ridx = ridx if type(ridx) == list else [ridx]
        assert type(ridx) == list
        for idx in ridx:
            print(f'{"":-<5}part: {part} idx: {idx} replaced')
            print('before: ', data[idx].strip("\n"))
            content = content if content.endswith('\n') else f'{content}\n'
            data[idx] = content
            print('after:  ', data[idx].strip("\n"))
        return data

    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.writelines(self.data)
            print(f'file is written to "{filepath}"')

    def __call__(self):
        print('\n')
        print(self.parts[0])


class CalDensity(object):
    def __init__(self):
        self.jiazhuangyoudongxi = None

    def __call__(self, dmf, K):
        Li7 = dmf.gt(k='3007.', p=2, im='s0').split()[-1]
        Li6 = dmf.gt(k='3006.', p=2, im='s0').split()[-1]
        Be9 = dmf.gt(k='4009.', p=2, im='s0').split()[-1]
        U238 = dmf.gt(k='92238.', p=2, im='s0').split()[-1]
        U235 = dmf.gt(k='92235.', p=2, im='s0').split()[-1]
        # Th232 = dmf.gt(k='90232.', p=2, im='s0').split()[-1]
        Uin = float(U238)+float(U235)
        # Thn=float(Th232)
        Thn = 0
        Li = float(Li7) + float(Li6)
        Be = float(Be9)
        U = Uin
        Th = Thn
        Zr = 0
        tot = Li + Be + U + Th + Zr
        Li1 = Li / tot
        Be1 = Be / tot
        U1 = U / tot
        Th1 = Th / tot
        Zr1 = Zr / tot
        denLi = Li1 * 25.94 / (2.3581 - 0.0004902 * K)
        denBe = Be1 * 47.01 / (1.972 - 0.0000145 * K)
        denU = U1 * 314 / (7.784 - 0.000992 * K)
        denTh = Th1 * 308 / (7.108 - 0.000759 * K)
        denZr = Zr1 * 167.22 / (4.4893 - 0.00107 * K)
        M = Li1 * 25.94 + Be1 * 47.01 + U1 * 314 + Th1 * 308 + Zr1 * 167.22
        den = M / (denLi + denBe + denU + denTh + denZr)
        return den

if __name__ == '__main__':
    filename = 'inputs/n100502'
    dmf = DmFile(filename)  # 定义对象
    cald = CalDensity()
    dmf.d(k='mt2', p='all', rmm='after', icld=False, im='sa', sep=None, k2=None)
    dmf.append(content='print', part='all')
    dmf.m(k='kcode', p=2, c='500000', rm='s1', im='sa')
    a = cald(dmf, K=900)
    b = '%.4f' %a
    dmf.m(k='1', p=0, c='-'+b, rm='s2', im='s1')
    dmf.save('n100502.txt')


    dmf.m(k='2', p=0, c='tmp=8.6170E-8', rm='srk', im='s1', rk='tmp')
    dmf.m(k='m2', p=2, c='6000.10c', rm='s1', im='sa')
    dmf.m(k='mt2', p=2, c='graph.10t', rm='s1', im='sa')
    dmf.save('n100502c.txt')



    filename = 'inputs/n100502'
    dmf = DmFile(filename)  # 定义对象
    cald = CalDensity()
    dmf.d(k='mt2', p='all', rmm='after', icld=False, im='sa', sep=None, k2=None)
    dmf.append(content='print', part='all')
    dmf.m(k='kcode', p=2, c='500000', rm='s1', im='sa')
    a = cald(dmf, K=900)
    b = '%.4f' % a
    dmf.m(k='1', p=0, c=str('-'+b), rm='s2', im='s1')
    dmf.save('n100502.txt')
    dmf.m(k='1', p=0, c='tmp=8.6170E-8', rm='srk', im='s1', rk='tmp')
    dmf.m(k='3007.90c', p=2, c='3007.10c', rm='s1', im='sa')
    dmf.m(k='3006.90c', p=2, c='3006.10c', rm='s0', im='sa')
    dmf.m(k='4009.90c', p=2, c='4009.10c', rm='s0', im='sa')
    dmf.m(k='92238.90c', p=2, c='92235.10c', rm='s0', im='sa')
    dmf.m(k='92235.90c', p=2, c='92235.10c', rm='s0', im='sa')
    dmf.m(k='9019.90c', p=2, c='9019.10c', rm='s0', im='sa')

    dmf.save('n100502t.txt')

    filename = 'inputs/n100502'
    dmf = DmFile(filename)  # 定义对象
    cald = CalDensity()
    dmf.d(k='mt2', p='all', rmm='after', icld=False, im='sa', sep=None, k2=None)
    dmf.append(content='print', part='all')
    dmf.m(k='kcode', p=2, c='500000', rm='s1', im='sa')
    a=cald(dmf, K=900)
    b = '%.4f' % a
    dmf.m(k='1', p=0, c=str('-'+b), rm='s2', im='s1')
    dmf.save('n100502.txt')
    a=cald(dmf, K=1000)
    b = '%.4f' % a
    dmf.m(k='1', p=0, c=str('-'+b), rm='s2', im='s1')
    dmf.save('n100502d.txt')



    # 生成.txt
    # dmf.d(k='mt2', p='all', rmm='after', icld=False, im='sa', sep=None, k2=None)
    # dmf.append(content='print', part='all')
    # dmf.save('n100502c.txt')

    # 生成c.txt



    # den = cald(Uin=, Thn=, K)


    # dmf.m(k='u=3', p=0, c='content', rm='s2', k2='cell')  # 推荐方式1：双关键词，idx_mode为sa，带ERROR
    # dmf.m(k='1', p=0, c='content', rm='s2', im='s1')  # 推荐方式2：单关键词，idx_mode为split后第i个


