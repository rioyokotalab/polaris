import logging
import time

from polaris import Polaris, STATUS_SUCCESS, Trials, params

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def pseudo_train(params):
    print('Rank: %s' % rank)
    lr_squared = (params['lr'] - 0.006) ** 2
    weight_decay_squared = (params['weight_decay'] - 0.02) ** 2
    loss = lr_squared + weight_decay_squared
    return {
        'loss':  loss,
        'status': STATUS_SUCCESS,
    }


if __name__ == '__main__':
    bounds = [
        params.Bounds('lr', 0.001, 0.01),
        params.Bounds('weight_decay', 0.0002, 0.04),
    ]
    trials = Trials(exp_key='this_is_test')
    logger = logging.getLogger(__name__)
    polaris = Polaris(
            pseudo_train, bounds, 'bayesian_opt',
            trials, 100, logger)
    best_params = polaris.run_parallel()
    print(best_params)
