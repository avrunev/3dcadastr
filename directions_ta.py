import pandas as pd
import math
import argparse
import numpy.matlib as ml
import numpy
import openpyxl


class Datum:
    def __init__(self, _params, _ci, _co, _mz):
        """
        :param _params:
        :param _ci:
        :param _co:
        :param _mz:
        :return:
        """
        self.params = _params
        self.ci = _ci
        self.co = _co
        self.mz = _mz
        self.ro = 206265
        self.m_l = _params.loc[2]["Значение"]
        self.m_b = _params.loc[3]["Значение"]
        self.m_g = _params.loc[4]["Значение"]
        # формируем общий список пунктов из ci и co
        self.ci.insert(3, "Опред.", [0] * self.ci.count()[0], True)
        self.co.insert(3, "Опред.", [1] * self.co.count()[0], True)
        self.co_cols = {}
        for i, p in enumerate(self.co.iterrows()):
            self.co_cols[p[0]] = i
        print(self.co_cols)
        self.c = pd.concat([ci, co])
        for i, s in enumerate(self.mz["Пункт установки тахеометра"]):
            for j in ["L", "g", "b", "A", ]:
                if pd.isna(self.mz.loc[i][j]):
                    self.mz.at[i, j] = 0
            if pd.isna(self.mz.loc[i]["Пункт установки тахеометра"]):
                self.mz.at[i, "Пункт установки тахеометра"] = self.mz.loc[i-1]["Пункт установки тахеометра"]
        self.mz["Пункт установки тахеометра"] = mz["Пункт установки тахеометра"].astype(int)
        self.mz["Пункт наведения"] = pd.to_numeric(self.mz["Пункт наведения"], downcast="signed")
        self.mz["b"] = pd.to_numeric(self.mz["b"], downcast="signed")
        self.mz["L"] = pd.to_numeric(self.mz["L"], downcast="signed")
        self.mz["g"] = pd.to_numeric(self.mz["g"], downcast="signed")
        self.mz["A"] = pd.to_numeric(self.mz["A"], downcast="signed")
        self.mz.insert(6, "Опред.", [0]*self.mz.count()[0], True)
        self.beta = []
        self.L = []
        self.gamma = []
        self.phi = []
        self.al = []
        for i, s in enumerate(self.mz["Пункт установки тахеометра"]):
            self.mz.at[i, "Опред."] = self.c.loc[self.mz.loc[i]["Пункт установки тахеометра"]]["Опред."] or \
                                      self.c.loc[self.mz.loc[i]["Пункт наведения"]]["Опред."]
            # считаем углы beta
            if not i:
                pass
            elif self.mz.loc[i]["Пункт установки тахеометра"] == self.mz.loc[i-1]["Пункт установки тахеометра"] and \
                    (self.mz.loc[i]["Опред."] or self.mz.loc[i-1]["Опред."]):
                self.beta.append((self.mz.loc[i-1]["Пункт установки тахеометра"],
                                  self.mz.loc[i-1]["Пункт наведения"],
                                  self.mz.loc[i]["Пункт наведения"]))
            # считаем измерения длин
            if self.mz.loc[i]["L"] and self.mz.loc[i]["Опред."]:
                self.L.append((self.mz.loc[i]["Пункт установки тахеометра"], self.mz.loc[i]["Пункт наведения"]))
            # считаем вертикальные углы
            if self.mz.loc[i]["g"] and self.mz.loc[i]["Опред."]:
                self.gamma.append((self.mz.loc[i]["Пункт установки тахеометра"], self.mz.loc[i]["Пункт наведения"]))
        print(self.mz)
        print(self.beta)
        print(self.L)
        print(self.gamma)
        self.n = len(self.beta)+len(self.L)+len(self.gamma)
        self.m = 3*self.co.count()[0]

    def X(self, i):
        return self.c.loc[i]["Xm"]*100

    def Y(self, i):
        return self.c.loc[i]["Ym"]*100

    def H(self, i):
        return self.c.loc[i]["Hm"]*100

    def beta_line(self, k, i, j):
        """
        коэффициенты для строки v_beta
        :param k:
        :param i:
        :param j:
        :return: словарь, где ключ это номер пункта и 0 - X, 1 - Y, только для определяемых пунктов
        """
        res = {(k, 0): self.a(k, j) - self.a(k, i) if self.c.loc[k]["Опред."] else 0,
               (k, 1): self.b(k, j) - self.b(k, i) if self.c.loc[k]["Опред."] else 0,
               (j, 0): self.a(j, k) if self.c.loc[j]["Опред."] else 0,
               (j, 1): self.b(j, k) if self.c.loc[j]["Опред."] else 0,
               (i, 0): -self.a(i, k) if self.c.loc[i]["Опред."] else 0,
               (i, 1): -self.b(i, k) if self.c.loc[i]["Опред."] else 0}
        return res

    def L_line(self, i, j):
        """
        коэффициенты для строки v_L
        :param i:
        :param j:
        :return: словарь, где ключ это номер пункта и 0 - X, 1 - Y, только для определяемых пунктов
        """
        res = {(j, 0): math.cos(datum.alpha(i, j)) if self.c.loc[j]["Опред."] else 0,
               (j, 1): math.sin(datum.alpha(i, j)) if self.c.loc[j]["Опред."] else 0,
               (i, 0): -math.cos(datum.alpha(i, j)) if self.c.loc[i]["Опред."] else 0,
               (i, 1): -math.sin(datum.alpha(i, j)) if self.c.loc[i]["Опред."] else 0}
        return res

    def gamma_line(self, i, j):
        """
        коэффициенты для строки v_gamma
        :param i:
        :param j:
        :return: словарь, где ключ это номер пункта и 0 - X, 1 - Y, 2 -H, только для определяемых пунктов
        """
        res = {(i, 0): self.c_(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 1): self.d(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 2): -self.e(j, i) if self.c.loc[i]["Опред."] else 0,
               (j, 0): -self.c_(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 1): -self.d(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 2): self.e(i, j) if self.c.loc[j]["Опред."] else 0}
        # С53=(х3-х5)*(н3-н5)
        return res

    def phi_line(self, i, j):
        """
        коэффициенты для строки v_phi
        :param i:
        :param j:
        :return: словарь, где ключ это номер пункта и 0 - X, 1 - Y, 2 -H, только для определяемых пунктов
        """
        res = {(i, 0): self.c_(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 1): self.d(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 2): -self.e(j, i) if self.c.loc[i]["Опред."] else 0,
               (j, 0): -self.c_(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 1): -self.d(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 2): self.e(i, j) if self.c.loc[j]["Опред."] else 0}
        # С53=(х3-х5)*(н3-н5)
        return res

    def al_line(self, i, j):
        """
        коэффициенты для строки v_a
        :param i:
        :param j:
        :return: словарь, где ключ это номер пункта и 0 - X, 1 - Y, 2 -H, только для определяемых пунктов
        """
        res = {(i, 0): self.c_(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 1): self.d(j, i) if self.c.loc[i]["Опред."] else 0,
               (i, 2): -self.e(j, i) if self.c.loc[i]["Опред."] else 0,
               (j, 0): -self.c_(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 1): -self.d(j, i) if self.c.loc[j]["Опред."] else 0,
               (j, 2): self.e(i, j) if self.c.loc[j]["Опред."] else 0}
        # С53=(х3-х5)*(н3-н5)
        return res

    def punct2col(self, p):
        """
        номер пункта ключ словаря предыдущих функций в номер столбца матрицы А
        :param p:
        :return:
        """
        return self.co_cols[p[0]]*3+p[1]

    def column_names(self) -> list:
        """
        список имен столбцов
        :return:
        """
        src = [[f'dX{i}', f'dY{i}', f'dH{i}'] for i in self.co_cols]
        return [item for sublist in src for item in sublist]

    def a(self, i, j):
        return self.ro*math.sin(self.alpha(i, j))/self.S(i, j)

    def b(self, i, j):
        return -self.ro*math.cos(self.alpha(i, j))/self.S(i, j)

    def c_(self, j, i):
        return self.ro*(self.H(j)-self.H(i))*(self.X(j)-self.X(i))/(self.S(i, j)*self.SS(i, j))

    def d(self, j, i):
        return self.ro*(self.H(j)-self.H(i))*(self.Y(j)-self.Y(i))/(self.S(i, j)*self.SS(i, j))

    def e(self, j, i):
        return self.ro*self.S(i, j)/self.SS(i, j)

    def rumb(self, i, j):
        """
        румб (i,j)
        :param i:
        :param j:
        :return:
        """
        at = math.atan((self.Y(j)-self.Y(i))/(self.X(j)-self.X(i)))
        return at if at >= 0 else math.pi+at

    def alpha(self, i, j):
        """
        переход от румба(i,j) к дирекционному углу (i,j)
        :param i:
        :param j:
        :return:
        """
        if self.X(j)-self.X(i) >= 0 and self.Y(j)-self.Y(i) >= 0:
            return self.rumb(i, j)
        if self.X(j)-self.X(i) < 0 <= self.Y(j)-self.Y(i):
            return math.pi - self.rumb(i, j)
        if self.X(j)-self.X(i) < 0 and self.Y(j)-self.Y(i) < 0:
            return math.pi + self.rumb(i, j)
        if self.X(j)-self.X(i) >= 0 > self.Y(j)-self.Y(i):
            return 2*math.pi - self.rumb(i, j)

    def S(self, i, j):
        return math.sqrt((self.X(j)-self.X(i))**2+(self.Y(j)-self.Y(i))**2)

    def SS(self, i, j):
        return (self.X(j)-self.X(i))**2+(self.Y(j)-self.Y(i))**2+(self.H(j)-self.H(i))**2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Параметры')
    parser.add_argument('--src_file', type=str, help='Файл исходных данных')
    args = parser.parse_args()
    if not args.src_file:
        print("Не указан файл исходных данных")
        exit(1)
    filename = args.src_file
    params = pd.read_excel(filename, "Параметры проекта", engine='openpyxl')
    ci = pd.read_excel(filename, "Координаты исходных пунктов", engine='openpyxl',
                       index_col="Название исходного пункта")
    co = pd.read_excel(filename, "Координаты определяемых пунктов", engine='openpyxl',
                       index_col="Название определяемого пункта")
    mz = pd.read_excel(filename, "Матрица запроектированных", engine='openpyxl')
    print(ci)
    print(co)
    datum = Datum(params, ci, co, mz)
    print(datum.m, datum.n)
    A = ml.zeros([datum.n, datum.m])
    labels = ['']*datum.n
    # Формирование матрицы А
    for ii, beta in enumerate(datum.beta):
        beta_line = datum.beta_line(*beta)
        labels[ii] = f'vb_{beta[0]}-{beta[1]}-{beta[2]}'
        for ind in beta_line:
            if beta_line[ind]:
                A[ii, datum.punct2col(ind)] = beta_line[ind]
    for ii, L in enumerate(datum.L):
        L_line = datum.L_line(*L)
        labels[ii+len(datum.beta)] = f'vL_{L[0]}-{L[1]}'
        for ind in L_line:
            if L_line[ind]:
                A[ii+len(datum.beta), datum.punct2col(ind)] = L_line[ind]
    for ii, gamma in enumerate(datum.gamma):
        gamma_line = datum.gamma_line(*gamma)
        labels[ii + len(datum.beta)+len(datum.L)] = f'vg_{gamma[0]}-{gamma[1]}'
        for ind in gamma_line:
            if gamma_line[ind]:
                A[ii+len(datum.beta)+len(datum.L), datum.punct2col(ind)] = gamma_line[ind]
    print(A)
    print(labels)
    # Формирование матрицы P
    P = ml.diag([1.]*datum.n)
    for ii, L in enumerate(datum.L):
        P[ii+len(datum.beta), ii+len(datum.beta)] = datum.m_b**2/datum.m_l**2
    for ii, gamma in enumerate(datum.gamma):
        P[ii+len(datum.beta)+len(datum.L), ii+len(datum.beta)+len(datum.L)] = datum.m_b**2/datum.m_g**2
    print(P)
    N = A.T*P*A
    print(N)
    Q = N.I
    print(Q)
    M = ml.zeros([datum.m//3, 4])
    for ii in range(M.shape[0]):
        M[ii, :] = [datum.m_b*math.sqrt(Q[3*ii, 3*ii]), datum.m_b*math.sqrt(Q[3*ii+1, 3*ii+1]),
                    datum.m_b*math.sqrt(Q[3*ii, 3*ii]+Q[3*ii+1, 3*ii+1]), datum.m_b*math.sqrt(Q[3*ii+2, 3*ii+2])]
    cols = datum.column_names()
    print(cols)
    dfA = pd.DataFrame(A, columns=cols)
    dfA['index'] = numpy.array(labels)
    dfA = dfA.set_index('index').round(4)
    print(dfA)
    dfN = pd.DataFrame(N, columns=cols)
    dfN['index'] = numpy.array(cols)
    dfN = dfN.set_index('index')
    print(dfN)
    dfQ = pd.DataFrame(Q, columns=cols)
    dfQ['index'] = numpy.array(cols)
    dfQ = dfQ.set_index('index')
    print(dfQ)
    dfM = pd.DataFrame(M, columns=['mX', 'mY', 'm', 'mH'])
    print(co.columns[0])
    dfM['index'] = co.index.values
    dfM = dfM.set_index('index')
    print(dfM)

    comps = filename.split('\\')
    comps[-1] = f'Результат{comps[-1]}'
    res_file_name = '\\'.join(comps)
    print(res_file_name)
    writer = pd.ExcelWriter(res_file_name, engine='openpyxl')
    dfA.to_excel(writer, sheet_name="A", startrow=2)
    dfN.to_excel(writer, sheet_name="N", startrow=2)
    dfQ.to_excel(writer, sheet_name="Q", startrow=2)
    dfM.to_excel(writer, sheet_name="M", startrow=2)
    writer.close()
    srcfile = openpyxl.load_workbook(res_file_name, read_only=False)
    sheetname = srcfile['A']
    sheetname['A1'] = str('Матрица коэффициентов уравнений поправок в измерения')
    sheetname = srcfile['N']
    sheetname['A1'] = str('Матрица коэффициентов нормальных уравнений')
    sheetname = srcfile['Q']
    sheetname['A1'] = str('Матрица весовых коэффициентов')
    sheetname = srcfile['M']
    sheetname['A1'] = str('СКО координат определяемых пунктов')
    srcfile.save(res_file_name)
