import pickle

from polaris.trials import Trials
from polaris.params import calc_params


class Polaris(object):

    def __init__(
            self, fun, space, algo, trials,
            max_evals=10, logger=None, debug=False):
        """
        Polaris base class.

        Parameters
        ----------
        fun : callable
            function which Polaris will evaluate
        space : list of polaris.params
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

        self.fun = fun
        self.space = space
        self.algo = algo
        self.trials = trials
        self.max_evals = max_evals
        self.logger = logger

        if type(trials) == 'Trials':
            self.is_parallel = False
        else:
            self.is_parallel = True

    def run(self):
        """
        Start evaluation up to max_evals count
        """

        if self.trials is None:
            self.trials = Trials()

        for eval_count in range(self.max_evals):
            params = calc_params(self.trials, algo=self.algo)
            exp_result = self.fun(params)
            self.trials.add(exp_result, params, eval_count)

        if self.debug:
            pickle.dump(self.trials._trials)
