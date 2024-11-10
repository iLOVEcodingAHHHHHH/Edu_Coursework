import pandas as pd

examp = pd.Series([1,2,3,4,5])
print(examp)


class Sample:
    def __init__(self,sample):
        x=0
        self.sample = sample
        self.sample_mean = sample.mean


abc = Sample(examp)
print(abc.sample_mean)
