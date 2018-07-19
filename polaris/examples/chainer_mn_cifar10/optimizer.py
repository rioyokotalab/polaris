from functools import partial
import random

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import TransformDataset, get_cifar10
from chainercv import transforms
import chainermn

import cv2 as cv
from mpi4py import MPI
import numpy as np
from polaris import STATUS_SUCCESS, STATUS_FAILURE

import models.resnet50
import models.lenet5

OPTIMIZERS = {
    'momentum_sgd': chainer.optimizers.MomentumSGD,
    'adam': chainer.optimizers.Adam,
    }

ARCHS = {
    'lenet5': models.lenet5.LeNet5,
    'resnet50': models.resnet50.ResNet50,
}


OMIT_ARG_PARAMS = [
    'lr_decay_rate',
    'lr_decay_epoch',
    'weight_decay',
]

mpi_comm = MPI.COMM_WORLD


def cv_rotate(img, angle):
    img = img.transpose(1, 2, 0) / 255.
    center = (img.shape[0] // 2, img.shape[1] // 2)
    r = cv.getRotationMatrix2D(center, angle, 1.0)
    img = cv.warpAffine(img, r, img.shape[:2])
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype(np.float32)
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0,
        crop_size=(32, 32), train=True):
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Color augmentation
    if train and pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))

    return img, label


def run(params, exp_info, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    chainer.cuda.cupy.random.seed(args.seed)

    # Prepare ChainerMN communicator.
    if args.communicator == 'naive':
        print("Error: 'naive' communicator does not support GPU.\n")
        exit(-1)

    comm = chainermn.create_communicator(args.communicator, mpi_comm)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Experiment Info: {}'.format(exp_info))
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    class_labels = 10
    model = L.Classifier(ARCHS[args.arch](class_labels))
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    arg_params = {i: params[i] for i in params if i not in OMIT_ARG_PARAMS}
    optimizer = chainermn.create_multi_node_optimizer(
            OPTIMIZERS[args.optimizer_name](**arg_params), comm)
    optimizer.setup(model)
    optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(params['weight_decay']))

    if comm.rank == 0:
        train, valid = get_cifar10(scale=255.)
        mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
        std = np.std([x for x, _ in train], axis=(0, 2, 3))
    else:
        train, valid, mean, std = None, None, None, None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    valid = chainermn.scatter_dataset(valid, comm, shuffle=True)
    mean = mpi_comm.bcast(mean, root=0)
    std = mpi_comm.bcast(std, root=0)

    train_transform = partial(
        transform, mean=mean, std=std, random_angle=args.random_angle,
        pca_sigma=args.pca_sigma, expand_ratio=args.expand_ratio,
        crop_size=args.crop_size, train=True)
    valid_transform = partial(transform, mean=mean, std=std, train=False)
    train = TransformDataset(train, train_transform)
    valid = TransformDataset(valid, valid_transform)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                 repeat=False, shuffle=True)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    if comm.rank == 0:
        # Some display and output extensions are necessary only for one worker.
        # (Otherwise, there would just be repeated outputs.)
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    if args.optimizer_name != 'adam':
        trainer.extend(extensions.ExponentialShift(
            'lr', params['lr_decay_rate']),
            trigger=(params['lr_decay_epoch'], 'epoch'))

    # Run the training
    trainer.run()

    if comm.rank == 0:
        print('========== Done ===========')

        loss = trainer.observation['validation/main/loss']
        loss = chainer.cuda.to_cpu(loss)

        if np.isnan(loss):
            return {
                'status': STATUS_FAILURE
            }
        else:
            return {
                'loss': loss,
                'status': STATUS_SUCCESS
            }
