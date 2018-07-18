import copy
import pickle
import os

from polaris.params import Domain
from polaris.rabbitmq import JobClient


class Polaris(object):

    def __init__(
            self, fn, bounds, algo, trials,
            max_evals=10, min_ei=1e-05, exp_key=None,
            logger=None, debug=False, args=None):
        """
        A client for the Polaris.

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
        max_evals : int
            the maximum number of evaluations
        min_ei : int
            the minimum expected improvement to continue experiments
        exp_key : str
            the expriment key to distinguish each experiments
        logger : Logger
            user logger object
        debug : bool
            make Polaris debug mode
        args : tuple
            args to pass fn
        """

        self.fn = fn
        self.bounds = bounds
        self.algo = algo
        self.trials = trials
        self.max_evals = max_evals
        self.min_ei = min_ei
        self.debug = debug
        self.args = args
        if exp_key is None:
            self.exp_key = fn.__name__
        else:
            self.exp_key = exp_key
        self.exp_info = {
            'exp_key': exp_key,
            'eval_count': 1,
        }

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
        Run an experiment sequentially with one process up to max_evals count.
        The expriment will early stop, if params become None
        """

        self.logger.info('Start searching...')
        for eval_count in range(1, self.max_evals+1):
            params = self.domain.predict(self.trials, self.min_ei)
            if params is None:
                break

            fn_params = copy.copy(params)

            if self.args is not None:
                exp_result = self.fn(fn_params, self.exp_info)
            else:
                exp_result = self.fn(fn_params, self.exp_info, *self.args)

            self.trials.add(exp_result, params, self.exp_info)
            self.exp_info['eval_count'] += 1

            self.logger.debug(fn_params)

            with open(f'{self.exp_key}.p', mode='wb') as f:
                pickle.dump(self.trials, f)

        return self.trials.best_params

    def run_parallel(self):
        """
        Run an experiment in parallel.

        Polaris use RabbitMQ to pass arguments from client to worker.
        You need to start at least one worker to start an experiment.
        """

        job_client = JobClient(self)
        job_client.start()
        return self.trials.best_params
