from trainer.process import Process
from trainer.reader import Reader
from trainer.util import rmse

reader = Reader("data.csv")
p = Process(reader)
p.getTrainData(0.04, 0.06)
fullData = p.trainDataFull
knn_eu = p.trainKNNEuc()
knn_cr = p.trainKNNCorr()
print("KNN-EU : ", rmse(knn_eu, fullData))
print("KNN-CR : ", rmse(knn_cr, fullData))