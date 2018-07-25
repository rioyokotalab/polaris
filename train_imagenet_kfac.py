#!/usr/bin/env python

from __future__ import print_function
import argparse
import json
import multiprocessing
import numpy as np
import os
import socket
import shutil
import sys

import chainer
import chainer.cuda
from chainer import training
from chainer.training import extension
from chainer.training import extensions

import chainermn


import models.alex as alex
import models.googlenet as googlenet
import models.googlenetbn as googlenetbn
import models.nin as nin
import models.resnet50 as resnet50
import models.resnet50_constantBN as resnet50_constantBN

import train_imagenet


def observe_hyperparam(name):
    def observer(trainer):
        return getattr(trainer.updater.get_optimizer('main'), name)
    return extensions.observe_value(name, observer)


class FacebookLRDecay(extension.Extension):

    def __init__(self, init_lr, ref_lr):
        self.ref_lr = ref_lr
        self.init_lr = init_lr

    def lr(self, epoch):
        if epoch < 5:
            _lr = (self.ref_lr - self.init_lr) * epoch / 5 + self.init_lr
        elif epoch < 30:
            _lr = self.ref_lr
        elif epoch < 60:
            _lr = 0.1 * self.ref_lr
        elif epoch < 80:
            _lr = 0.001 * self.ref_lr
        else:
            _lr = 0.0001 * self.ref_lr
        return _lr

    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        optimizer = trainer.updater.get_optimizer('main')
        lr = self.lr(epoch)
        setattr(optimizer, 'lr', lr)


class AdaHyperParam(extension.Extension):

    def __init__(self, init_lr, init_damping):
        self.init_lr = init_lr
        self.init_damping = init_damping
        self.iteration = 0

    def hyper_param(self, epoch, iteration):
        if iteration < 71:
            _damping = self.init_damping * (0.886 ** self.iteration)
        else:
            _damping = self.init_damping * 0.0314

        if epoch < 5:
            _lr = self.init_lr / 5
        elif epoch < 30:
            _lr = self.init_lr
        elif epoch < 60:
            _lr = 0.1 * self.init_lr
        elif epoch < 70:
            _lr = 0.01 * self.init_lr
        else:
            _lr = 0.001 * self.init_lr

        return _damping, _lr

    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        iteration = trainer.updater.iteration
        optimizer = trainer.updater.get_optimizer('main')
        damping, lr = self.hyper_param(epoch, iteration)
        setattr(optimizer, 'lr', lr)
        setattr(optimizer, 'damping', damping)


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    if not chainer.cuda.available:
        raise RuntimeError("ImageNet requires GPU support.")

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnet50_constantBN': resnet50_constantBN.ResNet50,
    }
    opts = {
        'kfac': chainer.optimizers.KFAC,
        'adam': chainer.optimizers.Adam,
        'sgd': chainer.optimizers.MomentumSGD,
        'rmsprop': chainer.optimizers.RMSprop,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet50')  # NOQA
    parser.add_argument('--batchsize', '-B', type=int, default=32)
    parser.add_argument('--epoch', '-E', type=int, default=10)
    parser.add_argument('--initmodel')
    parser.add_argument('--loaderjob', '-j', type=int)
    parser.add_argument('--mean', '-m', default='mean.npy')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--train_root', default='.')
    parser.add_argument('--val_root', default='.')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='flat')
    parser.add_argument('--loadtype', default='io')
    parser.add_argument('--iterator', default='thread')
    parser.add_argument('--optimizer', default='kfac')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--statistics', action='store_true', default=False)
    parser.add_argument('--facebook_decay', action='store_true', default=False)
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cov_ema_decay', type=float, default=0.99)
    parser.add_argument('--damping', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=5e-05)
    parser.add_argument('--config_out', default='config.json')
    parser.add_argument('--lr_warmup', default='gradual')
    parser.add_argument('--acc_iters', type=int, default=1)
    parser.add_argument('--nccl_comm_dim', type=int, default=1)
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # ======== Create communicator ========
    if args.optimizer == 'kfac':
        comm = chainermn.create_kfac_communicator(
            args.communicator, debug=args.debug, dim=args.nccl_comm_dim)
    else:
        comm = chainermn.create_communicator(
            args.communicator, dim=args.nccl_comm_dim)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    # ======== Create model ========
    model = archs[args.arch]()
    if args.arch == 'resnet50' and args.optimizer == 'kfac':
        model = archs['resnet50_constantBN']()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # ======== Copy model to GPU ========
    try:
        chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
        model.to_gpu()
    except chainer.cuda.cupy.cuda.runtime.CUDARuntimeError as e:
        print('ERROR at {}: GPU id: {}'.format(socket.gethostname(), device),
              file=sys.stderr)
        raise e

    # ======== Create dataset ========
    if args.loadtype == 'io':
        dataset_class = chainer.datasets.CroppingImageDatasetIO
    elif args.loadtype == 'mem':
        dataset_class = chainer.datasets.CroppingImageDataset
    else:
        raise ValueError('Unknown loadtype: {}'.format(args.loadtype))
    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(args.mean)
    if comm.rank == 0:
        train = chainer.datasets.read_pairs(args.train)
        val = chainer.datasets.read_pairs(args.val)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)
    train = dataset_class(
        train, args.train_root, mean, model.insize, model.insize)
    val = dataset_class(
        val, args.val_root, mean, model.insize, model.insize, False)

    # ======== Create iterator ========
    if args.iterator == 'process':
        # We need to change the start method of multiprocessing module if we
        # are using InfiniBand and MultiprocessIterator. This is because
        # processes often crash when calling fork if they are using Infiniband.
        # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
        multiprocessing.set_start_method('forkserver')
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, args.val_batchsize, n_processes=args.loaderjob, repeat=False,
            shuffle=False)
    elif args.iterator == 'thread':
        train_iter = chainer.iterators.MultithreadIterator(
            train, args.batchsize, n_threads=args.loaderjob)
        val_iter = chainer.iterators.MultithreadIterator(
            val, args.val_batchsize, n_threads=args.loaderjob, repeat=False,
            shuffle=False)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val, args.val_batchsize, repeat=False, shuffle=False)

    # ======== Create optimizer ========
    optimizer = opts[args.optimizer]()
    if args.optimizer == 'kfac':
        optimizer = chainer.optimizers.KFAC(
            lr=args.lr, momentum=args.momentum,
            cov_ema_decay=args.cov_ema_decay, damping=args.damping,
            acc_iters=args.acc_iters)
    elif args.optimizer == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    if comm.rank == 0 and args.optimizer == 'kfac':
        print('indices: {}'.format(optimizer.indices))

    # ======== Create updater ========
    updater = training.StandardUpdater(train_iter, optimizer, device=device)

    # ======== Create trainer ========
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    # ======== Extend trainer ========
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')
    # Create a multi node evaluator from an evaluator.
    evaluator = train_imagenet.TestModeEvaluator(
        val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)
    # Reduce the learning rate
    if args.optimizer == 'kfac':
        trainer.extend(AdaHyperParam(args.lr, args.damping),
                       trigger=(1, 'iteration'))
        #trainer.extend(extensions.ExponentialShift('lr', 0.01),
        #               trigger=(50, 'epoch'))
    if args.optimizer == 'sgd' and args.facebook_decay:
        batchsize = args.batchsize * comm.size
        init_lr = args.lr
        ref_lr = init_lr * 2
        trainer.extend(FacebookLRDecay(init_lr, ref_lr), trigger=(1, 'epoch'))
    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
        if args.optimizer == 'kfac':
            trainer.extend(observe_hyperparam('damping'), trigger=log_interval)
            report = ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
                      'main/accuracy', 'validation/main/accuracy', 'lr', 'damping']
        else:
            report = ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
                      'main/accuracy', 'validation/main/accuracy', 'lr']
        trainer.extend(extensions.PrintReport(report), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        if args.statistics:
            trainer.extend(extensions.ParameterStatistics(model))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if comm.rank == 0:
        hyperparams = optimizer.hyperparam.get_dict()
        for k, v in hyperparams.items():
            print('{}: {}'.format(k, v))

    # ======== Save configration ========
    config = {}
    config['args'] = vars(args)
    config['hyperparams'] = optimizer.hyperparam.get_dict()

    with open(os.path.join(args.out, args.config_out), 'w') as f:
        r = json.dumps(config)
        f.write(r)

    # Copy this file to args.out
    shutil.copy(os.path.realpath(__file__), args.out)

    trainer.run()


if __name__ == '__main__':
    main()
