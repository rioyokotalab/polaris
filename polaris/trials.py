from enum import Enum


class TrialStatus(Enum):
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2


class Trials(object):

    def __init__(self, exp_key=None):
        self._trials = []
        self._exp_key = exp_key

    def add(self, result, params, eval_count):
        t = {
            'result': result,
            'loss': result.get('loss', None),
            'exp_key': self._exp_key,
            'params': params,
            'eval_count': eval_count,
        }

        if result['status'] == TrialStatus.SUCCESS:
            t['status'] = TrialStatus.SUCCESS
        else:
            t['status'] = TrialStatus.FAILURE

        self._trials.append(t)
