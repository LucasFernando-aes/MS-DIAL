# Improving Transferability of Domain Adaptation Networks Through Domain Alignment Layers

Pytorch implementation of the SIBGRAPI 2021 paper entitled [_"Improving Transferability of Domain Adaptation Networks Through Domain Alignment Layers"_](https://arxiv.org/abs/2109.02693). In this paper, we focused on improving the transferability of deep learning (DL) models with simple and efficient network layers easily pluggable into the network backbones of existing multi-source unsupervised domain adaptation (MSDA) methods. More specifically, we proposed to embed a Multi-Source version of DomaIn Alignment Layers (MS-DIAL) at different levels of any given DL model.

This code is highly based on the [DARN repository](https://github.com/junfengwen/DARN) with the addition of the MS-DIAL module and its insertion algorithm. We also made some code refactoring in the domain adaptation models to enable the insertion of MS-DIAL layers.

## Environment

To reproducibility purposes, we use the same requirements of DARN repository, that is:
- Pytorch version 1.4.0 (consequently CUDA 10.1)
- Python 3.7.2

In particular, we build a docker container to run our experiments into a nvidia-docker environment. Our used DockerFile, build script (*build.sh*) and run script (*run.sh*) are available at the nvidia-docker_CUDA10.1 folder. Run script must be adjusted accordingly to your system to load the repository and dataset folders into the container environment. Once build and running the container just change dir to the repository and run the experiments.

`./build.sh`

`./run.sh`

`cd into/some/dir && ./digits.sh `

## Reproducing Our Results

In order to reproduce our experiments you first need to download the datasets and place them into the directory_path (by default *../datasets* from the repo folder) and, after, perform:
- *digits_prepro.py* code to preprocess the digit datasets into a npz file, if you want to run digits recognition experiments.
    - Also needs some extra packages, for instance *scipy*, *skimage* and *imageio*.
- *get_features.py* code to preprocess the data following the imagenet preprocess steps and get its feature vectors obtained from a pretrainet ResNet50 network, if you want to run Office-31 or Office-Home object recognitions experiments.
    - To run *get_feautres.py* some configuration parameters are needed, for instance `<dataset> --data_path <dataset_path> -d domain1 -d domain2 ... -d domain_n`.

In sequence, you need to run the respective shell script file depending on the experiment you are aiming to reproduce, for instance, *digits.sh* to run digits experiments; *office_home.sh* to run Office-Home experiments; and *office_31.sh* to run Office-31 experiments. The results (log file and test accuracy file) will be stored in the *./result* folder sorted by a incremental index. Finally, to get the average and the standard error over the n expriments, run the `mean_stderr.py` code with the `<dataset name> <method> <projection> <file prefix>` parameters.

By default, DARN method w/o MS-DIAL layers will be evaluated, but it can be changed by setting the parameters *--method* and *--dial-choice=DIAL*. More parameters can also be viewed through the `-h` parameter on the *main.py* file.

## Contact

If you have any doubts, feel free to contact me through the email e.lucas@unifesp.br

## Citation
```
@InProceedings{SIBGRAPI 2021 Silva,
    author = {L. F. A. {Silva} and D. C. G. {Pedronette} and F. A. {Faria} and J. P. {Papa} and J. {Almeida}},
    title = {Improving Transferability of Domain Adaptation Networks Through Domain Alignment Layers},
    pages = {1–8},
    booktitle = {2021 34th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
    address = {Gramado, RS, Brazil},
    month = {October 18–22},
    year = {2021},
    publisher = {{IEEE}}
}
```
