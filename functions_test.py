import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from pytictoc import TicToc
t = TicToc()


def disp(y, x, h_x):
    '''
    возвращает взвешенную диспресию от таргета после расщепления на две части
    y - целевая переменная, x - признак по которому делим,
    h_x - граница, по которой делим признак
    '''
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


def find_min_std_pos(y, x, min_x, max_x):
    '''
    Поиск элемента для разделения признака методом половинного деления.
    Возвращает границу для разделения
    y - целевая переменная, x - признак по которому делим,
    min_x, min_x -- значения признака нужны для рекурсии
    '''
    l = max_x - min_x
    center = (max_x + min_x) / 2
    delta = (l) / 5

    left_addition = disp(y, x, min_x + delta) - disp(y, x, min_x)
    right_addition = disp(y, x, max_x) - disp(y, x, max_x - delta)
    center_addition = disp(y, x, center + delta / 2) - disp(y, x, center - delta / 2)
    # print(left_addition)
    # print(right_addition)
    # print(center_addition)
    # print('------')
    if left_addition * center_addition < 0:
        return find_min_std_pos(y, x, min_x, center)
    if right_addition * center_addition < 0:
        return find_min_std_pos(y, x, center, max_x)
    # TODO: -30?
    if abs(center_addition) < 1e-30:
        return center
    return find_min_std_pos(y, x, center - l / 4, center + l / 4)


def find_multiple_min_std(y, X, coordinate):
    ''' инициализация find_min_std_pos для входящего признака.
    Возвращает гарницу разделения и значение дисперсии '''
    print('coordinate', str(coordinate))
    # TODO: i'll must think about sturcture input X
    x = X[:, coordinate]
    # TODO: delete nans it isn't good way
    x = x[~np.isnan(x)]
    h = find_min_std_pos(y, x, min(x), max(x))
    return h, disp(y, x, h)


#----Create data ----#
max_x = 5
n_points = 30
x = np.linspace(0,max_x,n_points)

y = np.sin(x)

X = pd.DataFrame({
    # 'x' : y,
    'x_l1' : shift(y,1, cval=np.nan),
    'x_l2' : shift(y,2, cval=np.nan),
    'x_l3' : shift(y,3, cval=np.nan),
    'x_l4' : shift(y,4, cval=np.nan)
})

# generate data for saving forest, where
# M -- depth tree (number of nodes),
# K -- number of used features (used for each split in tree),
# N -- number of trees
M = 3
K = X.shape[1]-1 # NOTE: why minus 1?
N = 10

array_coordinates = [[random.randint(0, K) for i in range(2**M)] for j in range(N)]
array_boundaries = [[None for i in range(2 ** M)] for j in range(N)]
array_sigma = [[None for i in range(2**M)] for j in range(N)]
array_points = [[None for i in range(2**M*2)] for j in range(N)]

# add target to data array for convenient splitting
Z = np.array(X)
Z = np.column_stack([Z, np.array(y)])

def process_node(t, n):
    '''
    функция записывает наблюдения для дерева, а именно:
    координату, массивы наблюдений слева и справа от границы h, саму границу и взвешенную дисперсию.
    t -- current tree number,
    n -- current node number,
    h -- split bound
    d -- dispersion
    '''
    subset = array_points[t][n]
    coordinate = array_coordinates[t][n]
    h, d = find_multiple_min_std(subset[:, -1], subset, coordinate)

    array_boundaries[t][n] = h
    array_sigma[t][n] = d

    x = subset[:, coordinate]
    array_points[t][2 * n] = subset[(x < h), :]
    array_points[t][2 * n + 1] = subset[(x >= h), :]


def init_tree(t):
    array_points[t][1] = Z
    for n in range(1, 2**(M)):
        process_node(t, n)


def init_all_trees():
    for t in range(N):
        init_tree(t)

init_all_trees()





def get_subset_and_n(t, S, h, n, obs):
    coordinate = array_coordinates[t][n]
    value = obs[coordinate]
    x = S[:, coordinate]
    if value > h:
        subset = S[x > h, :]
        return subset, 2*n
    else:
        subset = S[x <= h, :]
        return subset, 2*n + 1


def prediction_of_tree(t, H, obs):
    n = 1
    subset = Z
    for i in range(M):
        subset, n = get_subset_and_n(t, subset, H[n], n, obs)

    return np.average(subset[:, -1])





def log_p_for_point(sigma, y, y_mean):
    return -0.5*np.log(2*np.pi) - np.log(sigma) - ((y - y_mean)**2 / (2*sigma**2))

def log_p_for_array(sigma, Y, y_mean):
    sum(log_p_for_point(sigma, y, y_mean) for y in Y)

def log_p_for_boundary(sigma, Z, h, coordinate):
    left = Z[np.where(Z[:, coordinate]) < h]
    right = Z[np.where(Z[:, coordinate]) <= h]
    Y_left = left[: K]
    Y_right = right[: K]

    mean_left = np.average(Y_left)
    mean_right = np.average(Y_right)

    return log_p_for_array(sigma, Y_left, mean_left) + log_p_for_array(sigma, Y_right, mean_right)

def log_p_for_tree(t, H):
    sum(log_p_for_boundary(array_sigma[i], array_points[i], H[i], array_coordinates[i]) for i in range(1, 2**M))

def log_p_for_all_trees(H):
    sum(log_p_for_tree(t, H) for t in range(N))

f = log_p_for_all_trees

MC_steps = 100
trajectory_array = [None for i in range(MC_steps)]
print(trajectory_array)
T = 1
L = 20

def step_len(grad):
    return T / grad

def default_step_len():
    return np.sqrt(2**M - 1)

def one_step_candidate(x0, grad):
    sigma = step_len(grad) / default_step_len()
    return [random.gauss(x0[i], sigma) for i in range(K)]

def step_candidates(x0, grad):
    return [one_step_candidate(x0, grad) for i in range(L)]

def actual_step_len(x0, x1):
    return np.sqrt(sum(x0[i] - x1[i])**2 for i in range(2**M - 1))

def get_grad(x0, v0, candidates_array, candidate_values):
    return max(abs((v0 - candidate_values[i]) / actual_step_len(x0, candidates_array[i])) for i in range(L))


def MCMC_step(x0, v0, grad):
    candidate_array = step_candidates(x0, grad)
    values = [f(candidate_array[i]) for i in range(L)]
    new_grad = get_grad(x0, v0, candidate_array, values)
    new_candidates = enlarge_steps(x0, candidate_array, grad, new_grad)
    new_values = [f(new_candidates[i]) for i in range(L)]



# def divide(subset, h, coordinate, part):
#     x = subset[:, coordinate]
#     if part == 0:
#         return x[x < h]
#     else:
#         return x[x >= h]

# def get_path(n, res_list):
#     '''  возвращает путь к узлу дерева с номером n '''
#     if (n/2 == 0):
#         return res_list
#     return get_path(n/2, res_list.append(n / 2))

# def get_subset(subset, path):
#     if len(path) == 0:
#         return subset
#     h = array_boudaries[path[-1]]
#     coordinate = array_coordinates[path[-1]]
#     new_sub = divide(subset, h, coordinate, path[-2] % 2)
#     return get_subset(new_sub, path[:-1])


#
# def get_min_disps_for_all_trees():
#     ''' поиск глобальных минимумов для расщипления признаков, выбранных случаный образом'''
#
#     return [[find_multiple_min_std(y, np.array(X), array_coordinates[i][j]) for i in range(M-1)] for j in range(N-1)]
#
# a = get_min_disps_for_all_trees()
# print(a)
