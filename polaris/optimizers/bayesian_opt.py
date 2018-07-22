import numpy as np
import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm


def expected_improvement(x, model, lowest_loss):
    next_params = x.reshape(1, -1)
    mu, sigma = model.predict(next_params, return_std=True)

    with np.errstate(divide='ignore'):
        diff = lowest_loss - mu
        z = diff / sigma
        ei = (lowest_loss - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] == 0.0

    return -1 * ei


def calc_next_params(domain, trials):
    # At first time, get random params
    if len(trials) <= 5:
        next_params = {}
        random_result = domain.random()
        for index, fieldname in enumerate(domain.fieldnames):
            next_params[fieldname] = random_result[index]
        return [next_params]

    train_x, train_y = trials.get_train_data()

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=9)

    model.fit(train_x, train_y)

    lowest_loss = trials.lowest_loss

    best_ei = -np.Inf
    n_iter = 400
    best_xs = []

    for _ in range(0, n_iter):
        minimize_result = minimize(
                fun=expected_improvement,
                x0=domain.random(),
                bounds=domain.bounds,
                method='L-BFGS-B',
                args=(model, lowest_loss),
                )
        ei = -minimize_result.fun
        if ei > best_ei:
            best_ei = ei
            best_xs.insert(0, minimize_result.x)

    next_params_candidates = []
    for best_x in best_xs:
        next_params = {}
        for index, fieldname in enumerate(domain.fieldnames):
            next_params[fieldname] = best_x[index]
        next_params_candidates.append(next_params)

    return next_params_candidates
