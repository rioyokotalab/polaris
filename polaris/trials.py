import numpy as np

STATUS_FAILURE = 1
STATUS_SUCCESS = 2


class Trials(object):
    """
    A class for storing trials data.
    """

    def __init__(self):
        self.trials = []
        self.lowest_loss = np.inf

        self.best_params = None
        self.last_params = None

        # instance variables for optimizers
        self._train_x = None
        self._train_y = None

    def __len__(self):
        return len(self.trials)

    def get_train_data(self):
        """
        A method to get train data.
        The train data is arranged in alphabetical order of
        the name of parameters.
        """

        return self._train_x, self._train_y

    def add(self, result, params, exp_info=None):
        """
        A method to add each trial result
        """

        loss = result.get('loss', np.inf)

        t = {
            'result': result,
            'loss': loss,
            'params': params,
            'exp_key': exp_info['exp_key'],
            'eval_count': exp_info['eval_count'],
        }

        if result['status'] == STATUS_SUCCESS:
            t['status'] = STATUS_SUCCESS

            # Because the iteration order is not determinstic,
            # sort params dictionary by alphabetical order.
            train_x_row = []
            for key in sorted(params):
                train_x_row.append(params[key])

            if self._train_x is None:
                self._train_x = np.array([train_x_row])
            else:
                self._train_x = np.vstack((self._train_x, train_x_row))

            if self._train_y is None:
                self._train_y = np.array([loss])
            else:
                self._train_y = np.hstack((self._train_y, loss))
        else:
            t['status'] = STATUS_FAILURE

        # Update last_params
        last_params = []
        for k in sorted(params):
            last_params.append(params[k])
        self.last_params = np.array(last_params)

        # Set lowest loss to calculate expected improvements.
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.best_params = params

        self.trials.append(t)

        return
