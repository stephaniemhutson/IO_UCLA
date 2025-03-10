import pandas as pd
import matplotlib.pyplot as plt


class GPVEstimator:
    def __init__(self, bids):
        self._bids = bids
        self._std = {'b': bids.std()}
        self._count = {'b': bids.count()}
        self._max = bids.max()
        self._min = bids.min()
        self._bw = {'b': self.bandwidth(stage='b')}
        self.__g_hat = {}
        self.__f_hat = {}
        self.__G_hat = {}
        self.__F_hat = {}
        self.__rho = 2

    def bandwidth(self, stage):
        return 1.06 * self._std[stage] * self._count[stage] ** (-0.2)

    def kernel(self, u):
        u = abs(u)
        if u > 1:
            return 0
        else:
            return 35/32 * (1-u**2)**3

    def g(self, b):
        if not self.__g_hat.get(b):
            self.__g_hat[b] = 1/(self._count['b']*self._bw['b']) * sum(
                [self.kernel((bid-b)/self._bw['b']) for bid in self._bids]
            )
        return self.__g_hat[b]


    def G(self, b):
        if not self.__G_hat.get(b):
            val = 1/self._count['b'] * sum ([int(bid <= b) for bid in self._bids])
            self.__G_hat[b] = val
        return self.__G_hat[b]

    def _val(self, b):
        if (b >= self._min + self.__rho * self._bw['b'] / 2)  and (b <= self._max - self.__rho * self._bw['b'] / 2):
            return self._min + self.G(b)/self.g(b)
        else:
            return 10000000000

    def get_valuations(self):
        valuations = []
        for bid in self._bids:
            val = self._val(bid)
            if val < 10000000000:
                valuations.append(val)
        self._vals = valuations
        vals_series = pd.Series(valuations)
        self._std['v'] = vals_series.std()
        self._count['v'] = vals_series.count()
        self._bw['v'] = self.bandwidth(stage='v')


    def f(self, v):
        if not self.__f_hat.get(v):
            self.__f_hat[v] = 1 / (self._count['b'] * self._bw['v']) * sum(
                    [self.kernel((val-v)/self._bw['v']) for val in self._vals]
                )
        return self.__f_hat[v]

    def graph_valuations(self):
        self.get_valuations()
        for val in self._vals:
            self.f(val)
        f_hat = [(k, v) for k, v in self.__f_hat.items()]
        f_hat = sorted(f_hat, key=lambda x: x[0])
        plt.plot([val[0] for val in f_hat], [val[1] for val in f_hat])
        plt.suptitle("Estimated Valuations Distribution using GPV")
        plt.show()


auctions = pd.read_csv('./PS2/PS3_Data/PS3Data.csv')
bids = pd.concat([auctions['Bidder_1'], (auctions['Bidder_2'])])
estimator = GPVEstimator(bids)

estimator.graph_valuations()
