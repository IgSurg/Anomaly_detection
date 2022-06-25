import sys
import os
import warnings
warnings.filterwarnings('ignore')
# from tqdm import tqdm

import pandas as pd
import numpy as np
import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize

import matplotlib.pyplot as plt

cur_dir = os.path.abspath(os.curdir)


class Monitor:

    def __init__(self,server_name,dfbase,nwindow = 120,doverinterval = 2, debug = 0):
        self._dfcheck = None
        self.server_name = server_name
        self.dfbase = dfbase
        """
        dfbase - DataFrame
        Формат входных данных dfbase - DataFrame c индексом по времени и нормализованными к 100 значениями метрики
        пример:
        p_time 	                CPU
        2018-10-22 00:00:25 	60.63
        
        nwindow - размер окна сглаживания по количеству точек времени в dfbase
        doverinterval - размер доверительного интервала
        debug - если значение > 0, то отображается plt
        
        Результат: dataFrame верхнего и нижнего доверительного интервала self.upper_bond и self.lower_bond
        """

        # расчет доверительного интервала
        rolling_mean = self.dfbase.rolling(window=nwindow).mean()

        # строим и доверительные интервалы для сглаженных значений
        rolling_std = self.dfbase.rolling(window=nwindow).std()
        self.upper_bond = rolling_mean + doverinterval * rolling_std
        self.lower_bond = rolling_mean - doverinterval * rolling_std

        #####################################
        if debug > 0:
            plt.figure(figsize=(15, 5))
            plt.title("Moving average")
            plt.plot(rolling_mean, "g", label="Rolling mean trend")

            plt.plot(self.upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(self.lower_bond, "r--")
            plt.legend(loc="upper left")
            plt.grid(True)
            plt.show()



    def anomaly(self,dfcheck,debug = 0):
        """
        dfbase - DataFrame
        Формат входных данных dfbase - DataFrame c индексом по времени и нормализованными к 100 значениями метрики
                пример:
                p_time 	                CPU
                2018-10-22 00:00:25 	60.63

                Результат: dataFrame
                p_time 	                CPU     Anomal
                2018-10-22 00:00:25 	60.63   True|False
        """
        self._dfcheck = dfcheck
        self._dfcheck['Anomal'] = False

        for i in range(len(self._dfcheck)):
            t = self._dfcheck.index[i] - datetime.timedelta(7)
            max_value_mean = (self.upper_bond[self.upper_bond.index > t].values[0] + self.upper_bond[self.upper_bond.index < t].values[
                -1]) / 2
            min_value_mean = (self.lower_bond[self.lower_bond.index > t].values[0] + self.lower_bond[self.lower_bond.index < t].values[
                -1]) / 2

            anomal = np.hstack(((self._dfcheck.iloc[i, 0] > max_value_mean), (self._dfcheck.iloc[i, 0] < min_value_mean)))
            self._dfcheck.iloc[i, 1] = anomal.any()
            pass

            #####################################
            if debug > 0:
                plt.figure(figsize=(15, 5))
                plt.title("Moving average")
                plt.plot(self._dfcheck.iloc[:, 0], "g", label="Rolling")

                plt.legend(loc="upper left")
                plt.grid(True)
                plt.show()

        return self.server_name, self._dfcheck





def read_data(path,files,number):
    try:
        df = pd.read_csv(path + files[number], sep=',')
        print(df.shape)
    except FileNotFoundError:
        print('What are you doing?')
    else:
        pass
    finally:
        pass
    return df




def main():

    path1 = './data/'
    files1 = ['fada280-HIST.CSV','februs44_HIST.CSV','februs1006_HIST.CSV','hapi3_HIST.CSV',
         'SBT-OANIR-0084_HIST.CSV']
    number1 = 0
    files2 = ['test.csv']
# чтение данных из файлов
    df = read_data(path1, files1, number1)
    df_check = read_data(path1, files2, number1)

######  Нормализация данных ############################################################
# приведение к нормальной форме с индексом по времени df
    df['p_time'] = pd.to_datetime(df['Time'], unit='s')
    df.index = df['p_time']
    del df['p_time']
    del df['Time']
    print('Time delta is ', df.index.max() - df.index.min())

    # приведение к нормальной форме с индексом по времени df
    df_check['p_time'] = pd.to_datetime(df_check['Time'], unit='s')
    df_check.index = df_check['p_time']
    del df_check['p_time']
    del df_check['Time']
    print('Time delta is ', df_check.index.max() - df_check.index.min())

#####################################################################################

#создание монитора
    server1 = Monitor('test',df)

#определение аномалий
    server_name, df_check_anomaly = server1.anomaly(df_check)
    print(server_name)

####################################################################################
#    plt.figure(figsize=(15, 5))
#    plt.title("Moving average")
#    plt.plot(df_check_anomaly.iloc[:,0], "g", label="Rolling")

#    plt.legend(loc="upper left")
#    plt.grid(True)
#    plt.show()
    pass




if __name__=='__main__':
    main()

