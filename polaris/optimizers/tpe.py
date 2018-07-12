import numpy as np
import scipy as sp
import statsmodels as sm
from scipy.optimize import minimize

def objective_function(x, l, g):
    return -l.pdf(x)/g.pdf(x)

class tpe:
    def __init__(self, gamma=0.15, bw=None, n_min=None, bw_weight=1.0):
        """
        tpe suggest
        gamma: float
            threshhold which separete data for model l and model g
            (default) 0.15
        bw : array of params
            bandwidth of kernels
            (default) statsmodel's default params (scott's rule of thumb)
        n_min : int
            mininum number of data to run TPE
            (default) d (number of dimension) + 1
        bw_weight: float
            factor of l' model's multiplied bandwidth
            (default) 1.0
        """
        self.gamma = gamma
        self.bw = bw
        self.n_min = int(n_min)
        self.bw_weight = bw_weight
        pass

    def __call__(self, domain, trials):
        if len(trials) < self.n_min+2:
            return domain.random()

        train_x, train_y = trials.get_train_data()
        idx = np.argsort(train_y)
        n_min = self.n_min
        n = len(trials)
        l_len = max(n_min, int(n*self.gamma))
        g_len = max(n_min, n-l_len)
        x_l = train_x[idx[:l_len],:]
        x_g = train_x[idx[-g_len:],:]
#
#       I want to get the types of domain
#       v_types = "ccccuuuuuooooo" ref. http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html
#
        v_types = 'c'*domain.n_params
        l = sm.nonparametric.KDEMultivariate(
            x_l,
            v_types
        )
        g = xm.nonparametric.KDEMultivariate(
            g_l,
            v_types
        )
        # l_dash = sm.nonparametric.KDEMultivariate(
        #     x_l,
        #     v_types,
        #     bw = l.bw*self.bw_weight
        # )
        minimize_result = minimize(
            fun = objective_function,
            x0 = np.mean(x_l,axis=0),
            bounds = domain.bounds,
            method = 'L-BFGS-B',
            args = (l, g)
        )

        next_params = {}
        for index, fieldname in enumerate(domain.fieldnames):
            next_params[fieldname] = minimize_result.x[index]

        return next_params









def test():
    pass

if __name__ == '__main__':
    test()
