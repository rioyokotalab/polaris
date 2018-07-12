import pickle

from polaris.trials import Trials
from polaris.params import Domain


class Polaris(object):

    def __init__(
            self, fn, bounds, algo, trials,
            max_evals=10, logger=None, debug=False):
        """
        Polaris base class.

        Parameters
        ----------
        fn : callable
            fnction which Polaris will evaluate
        bounds : array of params
            search range of hyperparameters
        algo : str
            search algorithms (random, tpe, bayesian)
        trials : polaris.trials
            trials object which store all parameters of trials
        logger : Logger
            user logger object
        debug : bool
            make Polaris debug mode
        """

        self.fn = fn
        self.bounds = bounds
        self.algo = algo
        self.trials = trials
        self.max_evals = max_evals
        self.logger = logger
        self.debug = debug

        # if type(trials) == 'Trials':
        #     self.is_parallel = False
        # else:
        #     self.is_parallel = True

    def run(self):
        """
        Start evaluation up to max_evals count
        """

        domain = Domain(self.bounds, algo=self.algo)

        for eval_count in range(self.max_evals):
            params = domain.predict(self.trials)
            exp_result = self.fn(params)
            self.trials.add(exp_result, params, eval_count)

        # print(self.trials.best_params)

        if self.debug:
            pickle.dump(self.trials.trials)

        return self.trials.best_params

    def run_async(self, multi_node=False):
        print(self.fn.__name__)
