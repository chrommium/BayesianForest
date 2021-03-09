import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from pytictoc import TicToc
t = TicToc()


class BayesianForestRegression:
    def __init__(self, X, y, M=3, N=10, K=4):
        ''' M - depth tree (number of nodes),
            K - number of used features (used for each split in tree),
            N - number of trees '''
        self.X = X
        self.y = y
        self.M = M
        self.N = N
        self.K = K
        self.K = X.shape[1]-1 # NOTE: why minus 1?
        # create arrays of: random features, empty bouds, sigmas and split points
        self.n_of_nodes = (2**self.M)# - 1
        self.array_coordinates = [[random.randint(0, K) for i in range(self.n_of_nodes)] for j in range(N)]
        self.array_boundaries = [[None for i in range(self.n_of_nodes)] for j in range(N)]
        self.array_sigma = [[None for i in range(self.n_of_nodes)] for j in range(N)]
        self.array_points = [[None for i in range(self.n_of_nodes*2)] for j in range(N)]

        # add target to data array for convenient splitting
        Z = np.array(X)
        self.Z = np.column_stack([Z, np.array(y)])


    def disp(self, y, x, h_x):
        ''' возвращает взвешенную диспресию от таргета после расщепления на две части
            y - целевая переменная, x - признак по которому делим,
            h_x - граница, по которой делим признак '''
        # left_array = y[np.where(x <= h_x)]
        # right_array = y[np.where(x > h_x)]
        inds_left = [i for i in range(len(x)) if x[i] <= h_x]
        inds_right = [i for i in range(len(x)) if x[i] > h_x]
        left_array = [y[i] for i in inds_left]
        right_array = [y[i] for i in inds_right]

        d1 = np.std(left_array)
        d2 = np.std(right_array)
        l1 = len(left_array)
        l2 = len(right_array)
        if l1 <= 1:
            d1 = 0
        if l2 <= 1:
            d2 = 0
        return ((l1 * d1) ** 2) + ((l2 * d2) ** 2)


    def find_min_std_pos(self, y, x, min_x, max_x):
        ''' Поиск элемента для разделения признака методом половинного деления.
            Возвращает границу для разделения
            y - целевая переменная, x - признак по которому делим,
            min_x, min_x -- значения признака нужны для рекурсии '''
        l = max_x - min_x
        center = (max_x + min_x) / 2
        delta = (l) / 5

        left_addition = self.disp(y, x, min_x + delta) - self.disp(y, x, min_x)
        right_addition = self.disp(y, x, max_x) - self.disp(y, x, max_x - delta)
        center_addition = self.disp(y, x, center + delta / 2) - self.disp(y, x, center - delta / 2)

        if left_addition * center_addition < 0:
            return self.find_min_std_pos(y, x, min_x, center)
        if right_addition * center_addition < 0:
            return self.find_min_std_pos(y, x, center, max_x)
        # TODO: -30?
        if abs(center_addition) < 1e-30:
            return center
        return self.find_min_std_pos(y, x, center - l / 4, center + l / 4)


    def find_multiple_min_std(self, y, X, coordinate):
        ''' инициализация find_min_std_pos для входящего признака.
        Возвращает гарницу разделения и значение дисперсии '''
        print('coordinate', str(coordinate))
        # TODO: i'll must think about sturcture input X
        x = X[:, coordinate]
        # TODO: delete nans it isn't good way
        x = x[~np.isnan(x)]
        h = self.find_min_std_pos(y, x, min(x), max(x))
        return h, self.disp(y, x, h)

    
    def process_node(self, t, n):
        ''' функция записывает наблюдения для дерева, а именно:
            координату, массивы наблюдений слева и справа от границы h, саму границу и взвешенную дисперсию.
            t -- current tree number,
            n -- current node number,
            h -- split bound
            d -- dispersion '''
        subset = self.array_points[t][n]
        coordinate = self.array_coordinates[t][n]
        h, d = self.find_multiple_min_std(subset[:, -1], subset, coordinate)

        self.array_boundaries[t][n] = h
        self.array_sigma[t][n] = d

        x = subset[:, coordinate]
        self.array_points[t][2 * n] = subset[(x < h), :]
        self.array_points[t][2 * n + 1] = subset[(x >= h), :]


    def init_tree(self, t):
        self.array_points[t][1] = self.Z
        # NOTE: form one to 2^M or from zero to 2^M-1?
        for n in range(1, 2**(self.M)):
            self.process_node(t, n)


    def init_all_trees(self):
        for t in range(self.N):
            self.init_tree(t)

