from typing import Callable
import numpy as np

__all__ = ("Reader")

class Reader:
    def __init__(
        self,
        fileName : str = "data.csv",
        forwardFunc : Callable = np.log,
        backwardFunc : Callable  = np.exp
    ):
        self.forwardFunc : Callable = forwardFunc
        self.backwardFunc : Callable = backwardFunc
        data = []
        with open(fileName, "rt") as f:
            _, *self.labels = f.readline().strip().split(",")
            for line in f:
                _, *subData = line.strip().split(",")
                data.append([float(x) if x != "NA" else float("NAN") for x in subData])
        self.data : np.ndarray = np.array(data, dtype = float)
    def getData(self) -> np.ndarray :
        return self.forwardFunc(self.data)
    def retrieveData(self, data : np.ndarray) -> np.ndarray :
        return self.backwardFunc(data)

if __name__ == "__main__":
    r = Reader()
    print(r.data)
    print(r.labels)
