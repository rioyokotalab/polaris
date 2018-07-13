import copy
import pickle
import os

from polaris.params import Domain
from polaris.rabbitmq import JobClient


class Polaris(object):

    def __init__(
            self, fn, bounds, algo, trials,
            max_evals=10, exp_key=None, logger=None, debug=False):
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
-           make Polaris debug mode
        """

        self.fn = fn
        self.bounds = bounds
        self.algo = algo
        self.trials = trials
        self.max_evals = max_evals
        self.debug = debug

        if exp_key is None:
            self.exp_key = fn.__name__
        else:
            self.exp_key = exp_key

        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)

            if self.debug:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)
            stream = logging.StreamHandler()
            formatter = logging.Formatter(
                    '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
            stream.setFormatter(formatter)
            self.logger.addHandler(stream)
        else:
            self.logger = logger

        self.domain = Domain(self.bounds, algo=self.algo)

    def run(self):
        """
        Start evaluation up to max_evals count
        """

        self.logger.info('Start searching...')
        for eval_count in range(1, self.max_evals+1):
            params = self.domain.predict(self.trials)
            fn_params = copy.copy(params)
            fn_params['eval_count'] = eval_count
            exp_result = self.fn(fn_params)
            self.trials.add(exp_result, params, eval_count, self.exp_key)
            self.logger.debug(fn_params)

        with open(f'{self.exp_key}.p', mode='wb') as f:
            pickle.dump(self.trials.trials, f)

        return self.trials.best_params

    def run_parallel(self):
        job_client = JobClient(self)
        job_client.start()
        return self.trials.best_params
