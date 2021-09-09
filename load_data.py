import os

import torch
import torchvision
import torchvision.transforms as T

import numpy as np
from scipy.sparse import coo_matrix


def load_numpy_data(name,
                    data_path,
                    logger):

    if name == "digits":

        data_names = ["mnist", "mnist_m", "svhn", "synth_digits"]
        num_trains = 20000
        num_tests = 9000
        input_dim = 2304  # number of features after CovNet

        train_insts, train_labels, test_insts, test_labels = [], [], [], []
        for dataset in data_names:
            data = np.load(os.path.join(data_path,
                                        dataset,
                                        "%s.npz" % dataset))
            logger.info("%s with %d training and %d test instances" % (dataset,
                                                                       data['train_x'].shape[0],
                                                                       data['test_x'].shape[0]))
            # Shuffle and get training and test data
            ridx = np.arange(data['train_x'].shape[0])
            np.random.shuffle(ridx)
            train_insts.append(data['train_x'][ridx[:num_trains]])
            train_labels.append(data['train_y'][ridx[:num_trains]])

            ridx = np.arange(data['test_x'].shape[0])
            np.random.shuffle(ridx)
            test_insts.append(data['test_x'][ridx[:num_tests]])
            test_labels.append(data['test_y'][ridx[:num_tests]])

        configs = {"input_dim": input_dim,
                   "channels": 3,
                   "conv_layers": [64, 128, 256],
                   "cls_fc_layers": [2048, 1024],
                   "dom_fc_layers": [2048, 2048],
                   "num_classes": 10,
                   "drop_rate": 0.0}

    elif name == "office_home":

        data_names = ['Art', 'Clipart', 'Product', 'Real_World']
        num_trains = 2000

        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            data_preprocess_dir = os.path.join(data_path, 'office_home_' + dataset.lower() + '.npz')

            assert os.path.exists(data_preprocess_dir), "Resnet50 features must be obtained from get_features.py code"
            data = np.load(data_preprocess_dir)

            t_samples = data['train_x']
            v_samples = data['test_x']
            t_labels  = data['train_y']
            v_labels  = data['test_y']

            num_insts.append(t_labels.shape[0])
            logger.info("%s with %d instances." % (dataset, num_insts[i]))

            ridx = np.arange(num_insts[i])
            np.random.shuffle(ridx)

            train_insts.append(t_samples[ridx[:num_trains]])
            train_labels.append(t_labels[ridx[:num_trains]])
            test_insts.append(v_samples[ridx[num_trains:]])
            test_labels.append(v_labels[ridx[num_trains:]])

        configs = {"input_dim": 2048,
                   "hidden_layers": [1000, 500, 100],
                   "num_classes": 65,
                   "drop_rate": 0.7}

    elif name == "office31":

        data_names = ['Amazon', 'DSLR', 'Webcam']
        num_trains = 410

        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            data_preprocess_dir = os.path.join(data_path, 'office31_' + dataset.lower() + '.npz')
    
            assert os.path.exists(data_preprocess_dir), "office31 features must be obtained from get_features.py code"
            data = np.load(data_preprocess_dir)

            t_samples = data['train_x']
            v_samples = data['test_x']
            t_labels  = data['train_y']
            v_labels  = data['test_y']

            num_insts.append(t_labels.shape[0])
            logger.info("%s with %d instances." % (dataset, num_insts[i]))

            ridx = np.arange(num_insts[i])
            np.random.shuffle(ridx)

            train_insts.append(t_samples[ridx[:num_trains]])
            train_labels.append(t_labels[ridx[:num_trains]])
            test_insts.append(v_samples[ridx[num_trains:]])
            test_labels.append(v_labels[ridx[num_trains:]])

        configs = {"input_dim": 2048,
                   "hidden_layers": [1000, 500, 100],
                   "num_classes": 31,
                   "drop_rate": 0.7}

    else:

        raise ValueError("Unknown dataset.")

    return data_names, train_insts, train_labels, test_insts, test_labels, configs


def data_loader(inputs, targets, batch_size, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    inputs_size = inputs.shape[0]
    if shuffle:
        random_order = np.arange(inputs_size)
        np.random.shuffle(random_order)
        inputs, targets = inputs[random_order, :], targets[random_order]
    num_blocks = int(inputs_size / batch_size)
    for i in range(num_blocks):
        yield inputs[i * batch_size: (i+1) * batch_size, :], targets[i * batch_size: (i+1) * batch_size]
    if num_blocks * batch_size != inputs_size:
        yield inputs[num_blocks * batch_size:, :], targets[num_blocks * batch_size:]


def multi_data_loader(inputs, targets, batch_size, shuffle=True):
    """
    Both inputs and targets are list of numpy arrays, containing instances and labels from multiple sources.
    """
    assert len(inputs) == len(targets)
    input_sizes = [data.shape[0] for data in inputs]
    max_input_size = max(input_sizes)
    num_domains = len(inputs)
    if shuffle:
        for i in range(num_domains):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i] = inputs[i][r_order], targets[i][r_order]
    num_blocks = int(max_input_size / batch_size)
    for j in range(num_blocks):
        xs, ys = [], []
        for i in range(num_domains):
            ridx = np.random.choice(input_sizes[i], batch_size)
            xs.append(inputs[i][ridx])
            ys.append(targets[i][ridx])
        yield xs, ys


def loader_gen(loader, mode='inf'):
    # https://github.com/pytorch/pytorch/issues/1917#issuecomment-479482530
    while True:
        for images, targets in loader:
            yield images, targets
        if mode != 'inf':
            break
