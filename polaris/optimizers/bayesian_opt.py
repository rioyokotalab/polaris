import sklearn.gaussian_process as gp
from scipy.optimize import minimize


def expected_improvement(x, model, lowest_loss):
    next_params = x.reshape(1, -1)

    mu, sigma = model.predict(next_params, return_std=True)

    expected_lowest = mu - sigma
    expected_improvement = lowest_loss - expected_lowest

    return -1 * expected_improvement


def calc_next_params(domain, trials):
    next_params = {}

    # At first time, get random params
    if len(trials) <= 5:
        random_result = domain.random()
        for index, fieldname in enumerate(domain.fieldnames):
            next_params[fieldname] = random_result[index]
        return next_params

    train_x, train_y = trials.get_train_data()

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=9)

    model.fit(train_x, train_y)

    lowest_loss = trials.lowest_loss

    minimize_result = minimize(
            fun=expected_improvement,
            x0=trials.last_params,
            bounds=domain.bounds,
            method='L-BFGS-B',
            args=(model, lowest_loss)
            )

    for index, fieldname in enumerate(domain.fieldnames):
        next_params[fieldname] = minimize_result.x[index]

    return next_params
