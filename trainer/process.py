from os import read
from numpy.core.fromnumeric import shape
from scipy.fft import dct, idct
import numpy as np
from math import isnan
from random import sample
from tqdm import tqdm

from .reader import Reader
from .config import EPS
from .util import log, rmse, dropRowWithNAN, normalize, denormalize, lineOrdering, knnPredict

class Process:

    @log("read data and get training data without NAN")
    def __init__(self, reader, ncols = 10, K = 5):
        self.reader = reader
        trainDataFull = []
        self.fullData : np.ndarray = self.reader.getData()
        for row in self.fullData:
            row = row[:ncols]
            if not np.any(np.isnan(row)):
                trainDataFull.append(row)
        self.trainDataFullAll : np.ndarray = np.array(trainDataFull)
        self.knn_K = K

    @log("do system loss and random loss on train data")
    def getTrainData(self, systemLoss : float = 0.05, randomLoss : float = 0.05):
        trainData : np.ndarray = np.copy(self.trainDataFullAll)
        trainDataFull : np.ndarray = np.copy(self.trainDataFullAll)
        lossPos = [set() for _ in range(trainData.shape[1])]
        # system loss
        quantile = np.quantile(trainData, systemLoss, 0)
        for i in range(trainData.shape[0]):
            for j in range(trainData.shape[1]):
                if trainData[i][j] < quantile[j]:
                    lossPos[j].add(i)
        # random loss
        for j in range(trainData.shape[1]):
            remain = set(range(trainData.shape[0])) - lossPos[j]
            remain = sample(list(remain), int(trainData.shape[0] * randomLoss))
            for loss in remain:
                lossPos[j].add(loss)
        # do loss
        for j in range(trainData.shape[1]):
            for i in lossPos[j]:
                trainData[i][j] = float("nan")
        # delete row and line which is all-NAN
        for _ in range(2):
            trainDataFullList, trainDataList = [], []
            for i in range(trainData.shape[0]):
                if not np.all(np.isnan(trainData[i])):
                    trainDataFullList.append(trainDataFull[i])
                    trainDataList.append(trainData[i])
            trainDataFull = np.array(trainDataFullList, dtype = float).T
            trainData = np.array(trainDataList, dtype = float).T
        self.trainDataFull = trainDataFull
        self.trainData = trainData
        print(f"train data : {self.trainData.shape} with {np.sum(np.isnan(self.trainData))} NAN")

    @log("DCT is un-finish")
    def trainDCT(self) -> np.ndarray:
        dataFreq = []
        for j in range(self.trainData.shape[1]):
            subData = self.trainData[:, j]
            for i in range(subData.shape[0]):
                if isnan(subData[i]):
                    subData[i] = 0
            dataFreq.append(dct(subData, norm = "forward"))
        return idct(sum(dataFreq) / len(dataFreq), norm = "forward")

    @log("KNN-EU predict")
    def trainKNNEuc(self) -> np.ndarray:
        trainDataWithoutNAN = dropRowWithNAN(self.trainData)
        if trainDataWithoutNAN.size == 0:
            raise ValueError("NAN appears in each row")
        def weightFunc(arr1, arr2):
            return np.sum((arr1 - arr2) ** 2) ** 0.5
        ordering = lineOrdering(trainDataWithoutNAN, weightFunc)
        ret = np.copy(self.trainData)
        ret = knnPredict(ret, self.knn_K, ordering)
        return ret

    @log("KNN-CR or KNN-TN predict")
    def trainKNNCorr(self, trunc = False):
        trainData = np.copy(self.trainData)
        trainData, trainDataMean, trainDataStd = normalize(trainData, trunc)
        trainDataWithoutNAN = dropRowWithNAN(trainData)
        if trainDataWithoutNAN.size == 0:
            raise ValueError("NAN appears in each row")
        def weightFunc(arr1, arr2):
            corr = np.mean((arr1 - np.mean(arr1)) * (arr2 - np.mean(arr2))) / np.std(arr1) / np.std(arr2)
            if abs(corr) - 1 > EPS:
                raise ValueError("correlation >= 1")
            ret = 1 / (1 - abs(corr) + EPS)
            if corr < 0:
                ret = -ret
            return ret
        ordering = lineOrdering(trainDataWithoutNAN, weightFunc, reverse = True)
        ret = knnPredict(trainData, self.knn_K, ordering)
        ret = denormalize(ret, trainDataMean, trainDataStd)
        return ret
