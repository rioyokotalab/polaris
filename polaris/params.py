import numpy as np


from polaris.optimizers import bayesian_opt, rand, tpe

OPTIMIZERS = {
    'bayesian_opt': bayesian_opt.calc_next_params,
    'random': rand.calc_next_params
    'tpe': tpe.tpe()
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
        self.n_params = len(bounds)
        self._algo = algo
        self._bounds = sorted(bounds, key=lambda b: b.label)

    @property
    def bounds(self):
        array_bounds = []
        for b in self._bounds:
            array_bounds.append((b.low, b.high))
        return array_bounds

    @property
    def fieldnames(self):
        labels = []
        for b in self._bounds:
            labels.append(b.label)
        return labels

    def random(self):
        rand_param = []
        r = np.random.rand(1, self.n_params)
        for i, b in enumerate(self._bounds):
            p = b.low + r[0][i] * b.range()
            if b.q is None:
                rand_param.append(p)
            else:
                rand_param.append(round(p / b.q) * b.q)
        return np.array(rand_param)

    def predict(self, trials):
        return OPTIMIZERS[self._algo](self, trials)
