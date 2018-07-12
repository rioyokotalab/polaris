import copy
import pickle

from polaris.trials import Trials
from polaris.params import Domain
from polaris.rabbitmq import JobClient


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

        self.domain = Domain(self.bounds, algo=self.algo)

    def run(self):
        """
        Start evaluation up to max_evals count
        """

        for eval_count in range(self.max_evals):
            params = self.domain.predict(self.trials)
            fn_params = copy.copy(params)
            fn_params['eval_count'] = eval_count
            exp_result = self.fn(fn_params)
            self.trials.add(exp_result, params, eval_count)

        if self.debug:
            pickle.dump(self.trials.trials)

        return self.trials.best_params

    def run_parallel(self):
        job_client = JobClient(self)
        job_client.run()
        return self.trials.best_params
