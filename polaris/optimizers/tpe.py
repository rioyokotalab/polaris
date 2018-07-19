import numpy as np
import scipy as sp
import statsmodels.api as sm
# import statsmodels.api as sm

def objective_function(x, l, g):
    return -l.pdf(x)/g.pdf(x)


def trunc_range(a, b , m, s):
    # ref. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    return (a-m)/s,(b-m)/s

def minimize(
    fun,
    x,
    sampling_num,
    bw,
    bounds,
    args=None
):
    best = np.inf
    ret = None
    for p in x[np.random.choice(x.shape[0], sampling_num)]:
        sample = []
        for mean, sigma, (a, b) in zip(p, bw, bounds):
            a,b = trunc_range(a, b, mean, sigma)
            sample.append(sp.stats.truncnorm.rvs(a, b, loc=mean, scale=sigma))
            # ref. https://en.wikipedia.org/wiki/Truncated_normal_distribution
        val = fun(sample, *args)
        if not np.isfinite(val):
            continue
        if  val < best:
            best = val
            ret = sample
    return ret

# default parameter

default_gamma = 0.25
default_bw = None
default_n_min = 8
default_bw_weight = 3.
default_sampling_num = 64

# - default parameter

def calc_next_params(domain, trials):
    gamma = default_gamma
    bw = default_bw
    n_min = default_n_min
    bw_weight = default_bw_weight
    sampling_num = default_sampling_num

    next_params = {}
    if len(trials) <= n_min+2:
        random_result = domain.random()
        for index, fieldname in enumerate(domain.fieldnames):
            next_params[fieldname] = random_result[index]
        return next_params

    train_x, train_y = trials.get_train_data()
    idx = np.argsort(train_y)
    n = len(trials)
    l_len = max(n_min, int(n*gamma))
    g_len = max(n_min, n-l_len)
    x_l = train_x[idx[:l_len],:]
    x_g = train_x[idx[-g_len:],:]
#
#   I want to get the types of params from domain
#   v_types = "ccccuuuuuooooo" ref. http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html
#
    v_types = 'c'*domain.n_params
    l = sm.nonparametric.KDEMultivariate(
        x_l,
        v_types,
        bw=bw
    )
    g = sm.nonparametric.KDEMultivariate(
        x_g,
        v_types,
        bw=bw
    )

    wide_bw = l.bw*bw_weight
    for w in np.nditer(wide_bw,op_flags=['readwrite']):
        w[...] = max(w, 1e-3*bw_weight)
    bounds = domain.bounds
    minimize_result = minimize(
        fun=objective_function,
        x=x_l,
        sampling_num=sampling_num,
        bw=wide_bw,
        bounds=bounds,
        args=(l, g)
    )

    for index, fieldname in enumerate(domain.fieldnames):
        next_params[fieldname] = minimize_result[index]

    return next_params



def test():
    pass

if __name__ == '__main__':
    test()
