import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize


def expected_improvement(x, model, lowest_loss, n_params):
    next_params = x.reshape(-1, n_params)

    mu, sigma = model.predict(next_params, return_std=True)

    with np.errstate(divide='ignore'):
        Z = (mu - lowest_loss) / sigma
        expected_improvement = (mu - lowest_loss) * \
            norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def calc_next_params(domain, trials):
    # At first time, get random params
    if len(trials) == 0:
        return domain.random()

    train_x, train_y = trials.get_train_data()

    alpha = 1e-5
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(
            kernal=kernel,
            alpha=alpha,
            n_restarts_optimizer=10,
            normalize_y=True)

    model.fit(train_x, train_y)

    lowest_loss = trials.lowest_loss

    if trials.last_params is not None:
        last_params = trials.last_params
    else:
        last_params = domain.random()

    minimize_result = minimize(
            fun=expected_improvement,
            x0=last_params.reshape(1, -1),
            bounds=domain.bounds,
            method='L-BFGS-B',
            args=(model, lowest_loss, domain.n_params)
            )

    next_params = {}
    for index, fieldname in enumerate(domain.fieldnames):
        next_params[fieldname] = minimize_result.x[index]

    return next_params
