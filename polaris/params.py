import numpy as np

from optimizers import bayesian_opt

OPTIMIZERS = {
    'bayesian_opt': bayesian_opt.calc_next_params
}


class Bounds:

    def __init__(self, label, low, high, q=None):
        self.label = label
        self.low = low
        self.high = high
        self.q = q

    def range(self):
        return self.high - self.low


class Domain:

    def __init__(self, bounds, algo='random'):
        self._bounds = bounds
        self.n_params = len(bounds)

    @property
    def bounds(self):
        array_bounds = []
        for b in sorted(self._bounds, key=lambda b: b.label):
            array_bounds.append((b.low, b.high))
        return array_bounds

    @property
    def fieldnames(self):
        labels = []
        for b in sorted(self._bounds, key=lambda b: b.label):
            labels.append(b.label)
        return labels

    def random(self):
        rand_param = {}
        r = np.random.rand(1, self.n_params)
        for i, b in enumerate(self.bounds):
            label = b.label
            rand_param[label] = b.low + r[i] * b.range()
            if b.q is not None:
                rand_param[label] = round(rand_param[label] / b.q) * b.q
        return rand_param

    def predict(self, trials):
        # trialsの結果を元にpredictする
        return OPTIMIZERS[self.algo](self, trials)
