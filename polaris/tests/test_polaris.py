import logging
from unittest import TestCase

from polaris import Polaris, Status, Trials, params


def pseudo_train(params):
    lr_squared = (params['lr'] - 0.006) ** 2
    weight_decay_squared = (params['weight_decay'] - 0.02) ** 2
    loss = lr_squared + weight_decay_squared
    return {
        'loss':  loss,
        'status': Status.SUCCESS,
    }


class TestPolaris(TestCase):

    def test_polaris(self):
        bounds = [
            params.Bounds('lr', 0.001, 0.01),
            params.Bounds('weight_decay', 0.0002, 0.04),
        ]
        trials = Trials(exp_key='this_is_test')
        logger = logging.getLogger(__name__)
        polaris = Polaris(
                pseudo_train, bounds, 'tpe',
                trials, 100, logger)
        best_params = polaris.run()
        print(best_params)

        self.assertGreater(best_params['lr'], 0.005)
        self.assertLess(best_params['lr'], 0.007)

        self.assertGreater(best_params['weight_decay'], 0.01)
        self.assertLess(best_params['weight_decay'], 0.03)
