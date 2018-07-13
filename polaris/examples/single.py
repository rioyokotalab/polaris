from polaris import Polaris, Trials, params
from polaris.examples.utils import pseudo_train


if __name__ == '__main__':
    bounds = [
        params.Bounds('lr', 0.001, 0.01),
        params.Bounds('weight_decay', 0.0002, 0.04),
    ]
    trials = Trials()
    polaris = Polaris(
            pseudo_train, bounds, 'bayesian_opt',
            trials, max_evals=10, debug=True, exp_key='this_is_test')
    best_params = polaris.run()
    print('best params: ', best_params)
