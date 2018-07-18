# Polaris

Polaris is a hyperparamter tuning library focusing on reducing time.  
We plan to support every state-of-art tuning method as follows.

-  Random Search
-  Bayesian Optimization
-  Tree of Parzen Estimators (TPE)

## Prerequisites
- Python >= 3.6
- RabbitMQ

## Installation

```shell
$ brew install rabbitmq
$ pip install polaris
```

## Examples

### Sequential Execution

```python
from polaris import Polaris, STATUS_SUCCESS, Trials, Bounds


def pseudo_train(params, exp_info):
    lr_squared = (params['lr'] - 0.006) ** 2
    weight_decay_squared = (params['weight_decay'] - 0.02) ** 2
    loss = lr_squared + weight_decay_squared
    return {
        'loss':  loss,
        'status': STATUS_SUCCESS,
    }


if __name__ == '__main__':
    bounds = [
        Bounds('lr', 0.001, 0.01),
        Bounds('weight_decay', 0.0002, 0.04),
    ]
    trials = Trials()
    polaris = Polaris(
            pseudo_train, bounds, 'bayesian_opt',
            trials, max_evals=100, exp_key='this_is_test')
    best_params = polaris.run()
    print(best_params)
```

### Parallel Execution

#### Single Process

1. Run `rabbitmq-server`
1. Set `RABBIT_MQ_HOST`, `RABBIT_MQ_USERNAME`, `RABBIT_MQ_PASSWORD`
1. Run `polaris-worker --exp-key this_is_test`
1. Run codes as follows

### Multiple Processes (Use MPI)

1. Run `rabbitmq-server`
1. Set `RABBIT_MQ_HOST`, `RABBIT_MQ_USERNAME`, `RABBIT_MQ_PASSWORD`
1. Run `mpirun -n 4 polaris-worker --mpi --exp-key this_is_test`
1. Run codes as follows


```python

...

if __name__ == '__main__':
    bounds = [
        Bounds('lr', 0.001, 0.01),
        Bounds('weight_decay', 0.0002, 0.04),
    ]
    trials = Trials()
    polaris = Polaris(
            pseudo_train, bounds, 'bayesian_opt',
            trials, max_evals=100, exp_key='this_is_test')
    best_params = polaris.run_parallel()
    print(best_params)
```
