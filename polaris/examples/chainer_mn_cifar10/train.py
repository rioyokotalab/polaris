import argparse
import pickle

from polaris import Polaris, Trials, params

BOUNDS = {
    'momentum_sgd': [
            params.Bounds('weight_decay', 0, 0.001),
            params.Bounds('lr', 0.005, 0.02),
            params.Bounds('lr_decay_rate', 0.3, 0.7),
            params.Bounds('lr_decay_epoch', 30, 80, 10),
            params.Bounds('momentum', 0.6, 0.98),
        ],
    'adam': [
            params.Bounds('weight_decay', 0, 0.001),
            params.Bounds('alpha', 0, 0.005),
            params.Bounds('beta1', 0.6, 0.95),
            params.Bounds('beta2', 0.6, 0.9999),
            params.Bounds('eta', 0.9, 1.0),
            params.Bounds('eps', 1e-9, 1e-10),
        ],
    }

if __name__ == '__main__':
    import optimizer

    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')

    # Set model
    parser.add_argument('--arch', '-a', type=str, default='resnet50')

    # Set batchsize
    parser.add_argument('--batchsize', '-B', type=int, default=128)

    # Set running environment
    parser.add_argument('--communicator', type=str, default='pure_nccl')
    parser.add_argument('--epoch', '-e', type=int, default=250)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--early_stopping', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer_name', type=str)
    parser.add_argument('--use_mongo', action='store_true')

    # Data augmentation settings
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--pca_sigma', type=float, default=25.5)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])

    args = parser.parse_args()
    bounds = BOUNDS[args.optimizer_name]

    job_name = f'{args.arch}_cifar10_{args.optimizer_name}'
    trials = Trials()

    polaris = Polaris(
            optimizer.run, bounds, 'bayesian_opt',
            trials, max_evals=100, exp_key=job_name, args=(args,))

    best_params = polaris.run_parallel()
    print(best_params)