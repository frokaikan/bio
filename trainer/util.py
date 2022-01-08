from typing import List, Tuple, Callable
import numpy as np
from math import isnan
from time import time
from tqdm import tqdm
from functools import wraps

from .config import EPS, PRINT_LOG

__all__ = ("log", "rmse", "dropRowWithNAN", "normalize", "denormalize", "lineOrdering")

def log(info):
    def logWrapper(func):
        @wraps(func)
        def funcWrapper(*args, **kw):
            t = time()
            if PRINT_LOG:
                print(f"[INFO : {func.__name__}] ", info)
            ret = func(*args, **kw)
            if PRINT_LOG:
                print(f"[TIME : {func.__name__}] ", time() - t)
            return ret
        return funcWrapper
    return logWrapper

# @log("evaluate RMSE")
def rmse(pred : np.ndarray, real : np.ndarray) -> np.ndarray :
    return (np.sum((pred - real) ** 2) / pred.size) ** 0.5 / np.std(real)

# @log("drop NAN")
def dropRowWithNAN(arr : np.ndarray) -> np.ndarray:
    ret = []
    for row in arr:
        if not np.any(np.isnan(row)):
            ret.append(row)
    return np.array(ret, dtype = float)

# @log("normalize")
def normalize(arr : np.ndarray, trunc : bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    arrWithoutNAN = [[] for _ in range(arr.shape[1])]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not isnan(arr[i, j]):
                arrWithoutNAN[j].append(arr[i, j])
    mean = np.array([np.mean(_) for _ in arrWithoutNAN], dtype = float)
    std = np.array([np.std(_) for _ in arrWithoutNAN], dtype = float)
    if trunc:

        raise ValueError("KNN-TN is NOT support")
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - mean) / std
    return arr, mean, std

# @log("resume normalize")
def denormalize(arr : np.ndarray, mean : np.ndarray, std : np.ndarray) -> np.ndarray :
    for i in range(arr.shape[0]):
        arr[i] = arr[i] * std + mean
    return arr

def autoCompare(cls : type):
    def __ne__(x, y):
        return not x == y
    def __le__(x, y):
        return x == y or x < y
    def __gt__(x, y):
        return not x == y and not x < y
    def __ge__(x, y):
        return not x < y
    cls.__ne__ = __ne__
    cls.__le__ = __le__
    cls.__gt__ = __gt__
    cls.__ge__ = __ge__
    return cls

@autoCompare
class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def __eq__(self, other) -> bool:
        return abs(self.first - other.first) < EPS and abs(self.second - other.second) < EPS
    def __lt__(self, other) -> bool:
        if abs(self.first - other.first) > EPS:
            return self.first < other.first
        elif abs(self.second - other.second) > EPS:
            return self.second < other.second
        else:
            return False
    def __str__(self) -> str :
        return f"Pair({str(self.first)}, {str(self.second)})"

class Dist(Pair):
    def __init__(self, first, second):
        super().__init__(abs(first), second)
        self.weight = first
    def __str__(self):
        return f"Dist({str(self.first)}, {str(self.second)}, {str(self.weight)})"

# @log("compute KNN distance")
def lineOrdering(arr : np.ndarray, weightFunc : Callable, *, reverse = False):
    distPairs = [[] for _ in range(arr.shape[1])]
    for j in range(arr.shape[1]):
        for j2 in range(arr.shape[1]):
            if j != j2:
                w = weightFunc(arr[:, j], arr[:, j2])
                distPairs[j].append(Dist(w, j2))
    for _ in distPairs:
        _.sort(reverse = reverse)
    return distPairs

# @log("do KNN predict")
def knnPredict(arr : np.array, K : int, distPairs) -> np.ndarray:
    with tqdm(total = np.sum(np.isnan(arr))) as bar:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if isnan(arr[i, j]):
                    cnt = 0
                    allPairs = []
                    for pair in distPairs[j]:
                        if not isnan(arr[i, pair.second]):
                            allPairs.append(pair)
                            cnt += 1
                            if cnt == K:
                                break
                    weightSum = 0
                    pred = 0
                    for pair in allPairs:
                        weightSum += abs(pair.weight)
                        pred += arr[i, pair.second] * pair.weight
                    pred /= weightSum
                    arr[i, j] = pred
                    bar.update(1)
    return arr