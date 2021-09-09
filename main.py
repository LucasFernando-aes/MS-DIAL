import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap

from model import DarnMLP, DarnConv, MdmnMLP, MdmnConv, MsdaMLP, MsdaConv, MSDialMLP, MSDialConv, DIALInsertion
from load_data import load_numpy_data, data_loader, multi_data_loader
import utils

features = []
def get_activation():
    def hook(model, input, output):
        features.append(input[0].cpu().detach())
    return hook

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the dataset: [amazon|digits|office_home|office31].",
                    type=str, choices=['digits', 'office_home', 'office31'], default="digits")
parser.add_argument("--method", help="Choose a method: [darn|src|tar|dann|mdmn|msda|msdial].",
                    type=str, choices=['darn', 'src', 'tar', 'dann', 'mdmn', 'msda', 'msdial'], default="darn")
parser.add_argument("--result_path", help="Where to save results.",
                    type=str, default="./results")
parser.add_argument("--data_path", help="Where to find the data.",
                    type=str, default="../datasets")
parser.add_argument("--mode", help="Aggregation mode [dynamic|L2]: L2 for DARN, dynamic for MDAN.",
                    type=str, choices=['dynamic', 'L2'], default="L2")
parser.add_argument("--lr", help="Learning rate.",
                    type=float, default=1.0)
parser.add_argument("--mu", help="Hyperparameter of the coefficient for the domain adversarial loss.",
                    type=float, default=1e-2)
parser.add_argument("--gamma", help="Inverse temperature hyperparameter.",
                    type=float, default=0.1)
parser.add_argument("--msdial-use", dest='msdial_use', help="Whether to embed MS-DIAL layers",
                    action="store_true")
parser.add_argument("--msdial-weight", dest='msdial_weight', help="entropy weight on total loss",
                    type=float, default=0.001)
parser.add_argument("--epoch", help="Number of training epochs.",
                    type=int, default=50)
parser.add_argument("--batch_size", help="Batch size during training.",
                    type=int, default=128)
parser.add_argument("--cuda", help="Which cuda device to use.",
                    type=int, default=0)
parser.add_argument("--seed", help="Random seed.",
                    type=int, default=-1)
parser.add_argument("--dim-reduction", dest='reduction', help="Make dimensionality reduction by t-SNE or UMAP.",
                    type=str, default=None, choices=['tsne', 'umap'])
args = parser.parse_args()

#################### Path config ####################

result_path = os.path.join(args.result_path,
                           args.name,
                           args.method,
                           args.mode)

if not os.path.exists(result_path):
    os.makedirs(result_path)

logger_name = "gamma_%g_mu_%g_dial-weight_%g_seed_%d_lr_%g_epoch_%d_dial_%s" % (args.gamma,
                   args.mu,
                   args.msdial_weight,
                   args.seed,
                   args.lr,
                   args.epoch,
                   'MSDIAL' if args.msdial_use else 'None')

logger_dir = os.path.join(result_path, logger_name + '.0.log')

if os.path.isfile(logger_dir):
    f = filter(lambda x: x.startswith(logger_name) and x.endswith('.log'), os.listdir(result_path))
    f = map(lambda x: int(x.split(logger_name)[1].split('.')[1]), f)

    logger_dir = '.'.join([os.path.join(result_path, logger_name), str(max(f)+1), 'log'])

logger = utils.get_logger(logger_dir)

#################### Initial Configs ####################

logger.info("Hyperparameter setting = %s" % args)

device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size

if args.seed >= 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

#################### Loading the datasets ####################

time_start = time.time()

data_names, train_insts, train_labels, test_insts, test_labels, configs = load_numpy_data(args.name,
                                                                                          args.data_path,
                                                                                          logger)
configs["mode"] = args.mode
configs['mu'] = args.mu
configs["gamma"] = args.gamma
configs["dial_weight"] = args.msdial_weight if (args.msdial_use or args.method == 'msdial') else None
configs["num_src_domains"] = len(data_names) - 1
num_datasets = len(data_names)

logger.info("Time used to process the %s = %g seconds." % (args.name, time.time() - time_start))
logger.info("-" * 100)

test_results = {}
np_test_results = np.zeros(num_datasets)

#################### Model ####################

if args.method in ["dann", "src", "tar"]:
    # Combine all sources for these methods
    num_src_domains = configs["num_src_domains"] = 1
else:
    num_src_domains = configs["num_src_domains"]

logger.info("Model setting = %s." % configs)

#################### Train ####################

alpha_list = np.zeros([num_datasets, num_src_domains, args.epoch])

for i in range(num_datasets):

    # Build source instances
    source_insts = []
    source_labels = []
    for j in range(num_datasets):
        if j != i:
            source_insts.append(train_insts[j].astype(np.float32))
            source_labels.append(train_labels[j].astype(np.int64))

    # Build target instances
    target_insts = train_insts[i].astype(np.float32)
    target_labels = train_labels[i].astype(np.int64)

    # Model
    if args.method in ["darn", "dann", "src", "tar"]:

        if args.method == "dann" or args.method == "src":
            source_insts = [np.concatenate(source_insts, axis=0)]
            source_labels = [np.concatenate(source_labels)]
        elif args.method == "tar":
            source_insts = [target_insts]
            source_labels = [target_labels]

        if args.method == "src" or args.method == "tar":
            configs["mu"] = 0.

        if args.name in ["office_home", 'office31']:  # MLP
            model = DarnMLP(configs)
        elif args.name == "digits":  # ConvNet
            model = DarnConv(configs)

    elif args.method == "mdmn":

        if args.name in ["office_home", 'office31']:
            model = MdmnMLP(configs)
        elif args.name == "digits":
            model = MdmnConv(configs)

    elif args.method == "msda":

        if args.name in ["office_home", 'office31']:
            model = MsdaMLP(configs)
        elif args.name == "digits":
            model = MsdaConv(configs)

    elif args.method == "msdial":

        if args.name in ["office_home", 'office31']:
            model = MSDialMLP(configs)
        elif args.name == "digits":
            model = MSDialConv(configs)

        model = DIALInsertion(model, args.method, args.name, num_src_domains+1).to(device)

    else:
        raise ValueError("Unknown method")

    ## use msdial ?
    if args.msdial_use and args.method not in ['src', 'tar', 'msdial']:
        logger.info("Inserting MS-DIAL layers")
        model = DIALInsertion(model, args.method, args.name, num_src_domains+1).to(device)
    else:
        assert not (args.method in ['src', 'tar'] and args.msdial_use), 'MSDIAL layers can\'t be inserted on src and/or tar method.'
        logger.info("Not inserted MS-DIAL")
        model = model.to(device)

    #optim
    if args.method == 'msda' and not args.msdial_use:
        opt_G = optim.Adadelta(model.G_params, lr=args.lr)
        opt_C1 = optim.Adadelta(model.C1_params, lr=args.lr)
        opt_C2 = optim.Adadelta(model.C2_params, lr=args.lr)

    elif args.method == 'msda' and args.msdial_use:
        opt_G = optim.Adadelta(model.backbone.G_params, lr=args.lr)
        opt_C1 = optim.Adadelta(model.backbone.C1_params, lr=args.lr)
        opt_C2 = optim.Adadelta(model.backbone.C2_params, lr=args.lr)

    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Training phase
    model.train()
    time_start = time.time()
    for t in range(args.epoch):

        running_loss = 0.0
        train_loader = multi_data_loader(source_insts, source_labels, batch_size)

        for xs, ys in train_loader:

            for j in range(num_src_domains):

                xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)

            # can choose same batch of instances
            ridx = np.random.choice(target_insts.shape[0], batch_size)
            tinputs = target_insts[ridx, :]
            tinputs = torch.tensor(tinputs, requires_grad=False).to(device)

            if args.method != 'msda':
                optimizer.zero_grad()
                loss, alpha = model(xs, ys, tinputs)
                loss.backward()
                optimizer.step()
            else:  # special training step for msda
                loss = utils.msda_train_step(model, xs, ys, tinputs, opt_G, opt_C1, opt_C2)

            running_loss += loss.item()

        if args.method == 'mdmn' or (args.method == 'darn' and args.mode == 'L2'):

            logger.info("Epoch %d, Alpha on %s: %s" % (t, data_names[i], alpha))
            alpha_list[i, :, t] = alpha

        logger.info("Epoch %d, loss = %.6g" % (t, running_loss))

    logger.info("Finish training %s in %.6g seconds" % (data_names[i],
                                                        time.time() - time_start))

    #add forward hook to get features
    if args.reduction is not None and args.method != 'msda':
        features = []
        if args.msdial_use or args.method == 'msdial':
            model.backbone.class_net.final.register_forward_hook(get_activation())
        else:
            model.class_net.final.register_forward_hook(get_activation())
    elif args.reduction is not None and args.method == 'msda':
        features = []
        if args.msdial_use:
            model.backbone.class_net1.final.register_forward_hook(get_activation())
        else:
            model.class_net1.final.register_forward_hook(get_activation())

    model.eval()

    # Test (use another hold-out target)
    test_loader = data_loader(test_insts[i], test_labels[i], batch_size=1000, shuffle=False)
    test_acc = 0.
    with torch.no_grad():
        for xt, yt in test_loader:
            xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
            yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
            preds_labels = torch.squeeze(torch.max(model.inference(xt), 1)[1])
            test_acc += torch.sum(preds_labels == yt).item()
        test_acc /= test_insts[i].shape[0]
        logger.info("Test accuracy on %s = %.6g" % (data_names[i], test_acc))
        test_results[data_names[i]] = test_acc
        np_test_results[i] = test_acc

        if args.reduction is not None:
            features = torch.cat(features, dim=0).cpu().numpy()
            labels = test_labels[i]

            if args.reduction == 'tsne':
                tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                projetion = tsne.fit_transform(features)
            elif args.reduction == 'umap':
                umap_reducer = umap.UMAP()
                projection = umap_reducer.fit_transform(features)

            plt.figure(figsize=(16,10))
            sns_plot = sns.scatterplot(
                x='x_tsne', y='y_tsne',
                hue='Class',
                palette=sns.color_palette("hls", 10 if args.name == 'digits' else 65),
                data={'x_tsne':projection[:,0], 'y_tsne':projection[:,1], 'Class':labels},
                legend="full",
                alpha=0.8
            )

            plt.ylabel('')
            plt.xlabel('')
            plt.savefig("_".join([args.reduction, data_names[i], args.method, str(args.msdial_use), ".pdf"]), dpi=300)
            np.savez("_".join([args.reduction, data_names[i], args.method, str(args.msdial_use)]), x=projection[:, 0], y=projection[:, 1], labels=labels)

logger.info("All test accuracies: ")
logger.info(test_results)

# Save results to files
test_file = logger_dir.split('.log')[0] + '.test'
np.savetxt(test_file, np_test_results, fmt='%.6g')

if args.method == 'mdmn' or (args.method == 'darn' and args.mode == 'L2'):
    for i in range(num_datasets):
        alpha_file = logger_dir.split('.log')[0] + '.alpha-{}'.format(data_names[i])
        np.savetxt(alpha_file, alpha_list[i], fmt='%.6g')

logger.info("Done")
logger.info("*" * 100)
