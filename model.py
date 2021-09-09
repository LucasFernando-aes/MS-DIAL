import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dial import DIALBatchNorm3d, DIALBatchNorm2d, DIALBatchNorm1d

from module import L2ProjFunction, GradientReversalLayer
import utils

########## DIAL/autoDIAL components
class _FeatureAlignmentConvNd(nn.Module):

    __constants__ = ['num_domains', 'conv_dim']

    def __init__(self, num_domains, conv_dim, in_channels, out_channels, **kwargs):
        super(_FeatureAlignmentConvNd, self).__init__()
        self.num_domains = num_domains
        self.conv_dim = conv_dim

        if conv_dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
            self.bn = DIALBatchNorm1d(num_domains, out_channels)
        elif conv_dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
            self.bn = DIALBatchNorm2d(num_domains, out_channels)
        elif conv_dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
            self.bn = DIALBatchNorm3d(num_domains, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def extra_repr(self):
        s = ('num_domains={num_domains}, conv_dim={conv_dim}')
        return s.format(**self.__dict__)


class _FeatureAligmentLinear(nn.Module):

    __constants__ = ['num_domains']

    def __init__(self, num_domains, in_features, out_features, **kwargs):
        super(_FeatureAligmentLinear, self).__init__()
        
        self.num_domains = num_domains

        self.fc = nn.Linear(in_features, out_features, **kwargs)
        self.bn = DIALBatchNorm1d(num_domains, out_features)


    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x

    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

class DIALInsertion(nn.Module):
    __constants__ = ['num_domains']

    def __init__(self, backbone, method, dataset, num_domains, **kwargs):
        super(DIALInsertion, self).__init__()

        self.backbone = backbone
        self.num_domains = num_domains
        self.src_domains = num_domains - 1

        if method in ['darn', 'dann', 'src', 'tar', 'mdmn', 'msdial']:
            if dataset == 'office_home' or dataset == 'amazon' or dataset == 'office31':
                submodel_names = ['feature']
                submodels = [self.backbone.feature_net]
            elif dataset == 'digits':
                submodel_names = ['feature', 'class']
                submodels = [self.backbone.feature_net, self.backbone.class_net.hiddens]
        elif method == 'msda':
            if dataset == 'office_home' or dataset == 'amazon' or dataset == 'office31':
                submodel_names = ['feature']
                submodels = [self.backbone.feature_net]
            elif dataset == 'digits':
                submodel_names = ['feature']
                submodels = [self.backbone.feature_net]

        for n, submodel in zip(submodel_names, submodels):

            has_bn = False
            for name, layer in submodel.named_modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    has_bn = True
                    break

            if has_bn:
                dial = dict()
                for name, layer in self.backbone.named_modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        if isinstance(layer, nn.BatchNorm1d):
                            dial_bn = DIALBatchNorm1d(self.num_domains, layer.num_features, **kwargs)
                        elif isinstance(layer, nn.BatchNorm2d):
                            dial_bn = DIALBatchNorm2d(self.num_domains, layer.num_features, **kwargs)
                        elif isinstance(layer, nn.BatchNorm3d):
                            dial_bn = DIALBatchNorm3d(self.num_domains, layer.num_features, **kwargs)
                    
                        state_dict = layer.state_dict()

                        for i in range(self.num_domains):
                            dial_bn.bn[i].load_state_dict(state_dict, strict=False)

                        if 'weight' in state_dict.keys():
                            dial_bn.gamma.data.copy_(state_dict['weight'].data)
                        if 'bias' in state_dict.keys():
                            dial_bn.beta.data.copy_(state_dict['bias'].data)

                        dial[name] = dial_bn
            else:
                dial = dict()
                for name, layer in submodel.named_modules():
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        attr = {'in_channels':layer.in_channels, 'out_channels':layer.out_channels,
                                'kernel_size':layer.kernel_size, 'stride':      layer.stride,
                                'padding':    layer.padding,     'dilation':    layer.dilation,
                                'groups':     layer.groups}

                        if isinstance(layer, nn.Conv1d):
                            dial[name] = _FeatureAlignmentConvNd(self.num_domains,
                                                                1,
                                                                **attr)
                        elif isinstance(layer, nn.Conv2d):
                            dial[name] = _FeatureAlignmentConvNd(self.num_domains,
                                                                2,
                                                                **attr)
                        elif isinstance(layer, nn.Conv3d):
                            dial[name] = _FeatureAlignmentConvNd(self.num_domains,
                                                                3,
                                                                **attr)

                        state_dict = layer.state_dict()

                        if 'weight' in state_dict.keys():
                            dial[name].conv.weight.data.copy_(state_dict['weight'].data)
                        if 'bias' in state_dict.keys():
                            dial[name].conv.bias.data.copy_(state_dict['bias'].data)

                    elif isinstance(layer, nn.Linear):
                        attr = {'in_features':layer.in_features, 'out_features':layer.out_features}
                        dial[name] = _FeatureAligmentLinear(self.num_domains,
                                                        **attr)

                        state_dict = layer.state_dict()

                        if 'weight' in state_dict.keys():
                            dial[name].fc.weight.data.copy_(state_dict['weight'].data)
                        if 'bias' in state_dict.keys():
                            dial[name].fc.bias.data.copy_(state_dict['bias'].data)

            for key, value in dial.items():
                utils.rsetattr(submodel, key, value)

        if method == 'msda':
            utils.rsetattr(self.backbone, 'G_params', self.backbone.feature_net.parameters())
            utils.rsetattr(self.backbone, 'C1_params', self.backbone.class_net1.parameters())
            utils.rsetattr(self.backbone, 'C2_params', self.backbone.class_net2.parameters())


    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

    def forward(self, sinputs, soutputs, tinputs):
        return self.backbone(sinputs, soutputs, tinputs)

    def inference(self, x):
        return self.backbone.inference(x)

########## Some components ##########
class MLPNet(nn.Module):

    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]

        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        if self.process_final:
            return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return self.final(x)


class ConvNet(nn.Module):

    def __init__(self, configs):
        """
        Feature extractor for the image (digits) datasets
        """

        super().__init__()
        self.channels = configs["channels"]  # number of channels
        self.num_conv_layers = len(configs["conv_layers"])
        self.num_channels = [self.channels] + configs["conv_layers"]
        # Parameters of hidden, cpcpcp, feature learning component.
        self.convs = nn.ModuleList([nn.Conv2d(self.num_channels[i],
                                              self.num_channels[i+1],
                                              kernel_size=3)
                                    for i in range(self.num_conv_layers)])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability

    def forward(self, x):

        dropout = self.dropout
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(dropout(x))), 2, 2, ceil_mode=True)
        x = x.view(x.size(0), -1)  # flatten
        return x


########## Models ##########
# DARN and MDAN
class DarnBase(nn.Module):

    def __init__(self, configs):
        """
        Domain AggRegation Network.
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        self.mode = mode = configs["mode"]
        self.mu = configs["mu"]
        self.gamma = configs["gamma"]
        self.dial_weight = configs["dial_weight"]

        if mode == "L2":
            self.proj = L2ProjFunction.apply
        else:
            self.proj = None

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        batch_size = tinputs.shape[0]
        num_src_domains = len(sinputs)
        if len(tinputs.shape) == 2: #features
            img_shape = [tinputs.shape[-1]]
        else: #img
            img_shape = tinputs.shape[-3:]
        
        # Compute features
        input_feature = torch.stack((tinputs, *sinputs), dim=1).view((-1, *img_shape))
        features = self.feature_net(input_feature)
        st_features = features.view((batch_size, 1+num_src_domains, -1))
        t_features = st_features[:, 0, ...]
        s_features = st_features[:, 1:, ...]

        # Classification probabilities on k source domains.
        if self.dial_weight is not None:
            input_class = features
            logprobs = F.log_softmax(self.class_net(input_class), dim=1).view((batch_size, num_src_domains+1, -1)) 
            train_losses = [F.nll_loss(logprobs[:, i, ...], soutputs[i-1]) for i in range(1, num_src_domains+1)]
            train_losses = torch.stack(train_losses)

            target_loss = utils.entropy(logprobs[:, 0, ...])
        else:
            input_class = torch.reshape(s_features, (batch_size*num_src_domains, -1))
            logprobs = F.log_softmax(self.class_net(input_class), dim=1).view((batch_size, num_src_domains, -1))
            train_losses = [F.nll_loss(logprobs[:, i, ...], soutputs[i]) for i in range(num_src_domains)]
            train_losses = torch.stack(train_losses)

        # Domain classification accuracies.
        slabels = torch.ones([batch_size, 1], requires_grad=False,
                             dtype=torch.float32, device=tinputs.device)
        tlabels = torch.zeros([batch_size, 1], requires_grad=False,
                              dtype=torch.float32, device=tinputs.device)

        domain_losses = []
        for i in range(self.num_src_domains):
            input_domain = torch.stack((t_features, s_features[:, i, ...]), dim=1).view((-1, *t_features.shape[1:]))
            domain_out = self.domain_nets[i](self.grl(input_domain))
            domain_out = domain_out.view(batch_size, 2, -1)
            domain_losses.append( F.binary_cross_entropy_with_logits(domain_out[:, 1, ...], slabels) +
                                  F.binary_cross_entropy_with_logits(domain_out[:, 0, ...], tlabels) )
        domain_losses = torch.stack(domain_losses)

        loss, alpha = self._aggregation(train_losses, domain_losses)
        if self.dial_weight is not None:
            return loss + self.dial_weight * target_loss, alpha
        else:
            return loss, alpha

    def _aggregation(self, train_losses, domain_losses):
        """
        Aggregate the losses into a scalar
        """

        mu, alpha = self.mu, None
        if self.num_src_domains == 1:  # dann
            loss = train_losses + mu * domain_losses
        else:
            mode, gamma = self.mode, self.gamma
            if mode == "dynamic":  # mdan
                g = (train_losses + mu * domain_losses) * gamma
                loss = torch.logsumexp(g, dim=0) / gamma
            elif mode == "L2":  # darn
                g = gamma * (train_losses + mu * domain_losses)
                alpha = self.proj(g)
                loss = torch.dot(g, alpha) + torch.norm(alpha)
                alpha = alpha.cpu().detach().numpy()
            else:
                raise NotImplementedError("Unknown aggregation mode %s" % mode)

        return loss, alpha

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class DarnMLP(DarnBase):

    def __init__(self, configs):
        """
        DARN with MLP
        """

        super().__init__(configs)

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}
        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])

        self.domain_nets = nn.ModuleList([nn.Linear(configs["hidden_layers"][-1], 1)
                                          for _ in range(self.num_src_domains)])


class DarnConv(DarnBase):

    def __init__(self, configs):
        """
        DARN with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.class_net = MLPNet(cls_configs)

        dom_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["dom_fc_layers"],
                       "output_dim": 1,
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.domain_nets = nn.ModuleList([MLPNet(dom_configs)
                                          for _ in range(self.num_src_domains)])


# MDMN
class MdmnBase(nn.Module):

    def __init__(self, configs):
        """
        MDMN model
        """

        super().__init__()

        self.num_src_domains = configs["num_src_domains"]
        self.num_domains = configs["num_src_domains"] + 1
        self.mu = configs["mu"]
        self.dial_weight = configs['dial_weight']
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        batch_size = tinputs.shape[0]
        if len(tinputs.shape) == 2: #features
            img_shape = [tinputs.shape[-1]]
        else: #img
            img_shape = tinputs.shape[-3:]

        # Compute features
        input_features = torch.stack((tinputs, *sinputs), dim=1).view(-1, *img_shape)
        features = self.feature_net(input_features)
        st_features = features.view((batch_size, self.num_domains, -1))
        t_features = st_features[:, 0, ...]
        s_features = st_features[:, 1:, ...]

        # These will be used later to compute the gradient penalty
        src_rand = s_features[:, int(np.random.choice(self.num_src_domains, 1)), ...]
        epsilon = np.random.rand()
        interpolated = epsilon * src_rand + (1 - epsilon) * t_features

        # Classification probabilities on k source domains.
        if self.dial_weight is not None:
            input_classifier = features
            logprobs = F.log_softmax(self.class_net(input_classifier), dim=1).view((batch_size, self.num_domains, -1))
            train_losses = [F.nll_loss(logprobs[:, i, ...], soutputs[i-1]) for i in range(1, self.num_domains)]
            train_losses = torch.stack(train_losses)

            target_loss = utils.entropy(logprobs[:, 0, ...])
        else:
            input_classifier = torch.reshape(s_features, (batch_size*self.num_src_domains, -1))
            logprobs = F.log_softmax(self.class_net(input_classifier), dim=1).view((batch_size, self.num_src_domains, -1))
            train_losses = [F.nll_loss(logprobs[:, i, ...], soutputs[i]) for i in range(self.num_src_domains)]
            train_losses = torch.stack(train_losses)

        # Domain classification accuracies.
        input_domains = features
        pred = self.domain_net(self.grl(input_domains))
        inter_f = self.domain_net(interpolated)


        d_idx = np.concatenate([np.ones(batch_size, dtype=int) * i for i in range(self.num_domains)])
        # convert to one-hot
        d = np.zeros((d_idx.size, self.num_domains), dtype=np.float32)
        d[np.arange(d_idx.size), d_idx] = 1
        # compute weights
        weights, alpha = utils.compute_weights(d, pred.cpu().detach().numpy(), batch_size)
        weights = torch.from_numpy(weights).float().to(pred.device).detach()

        # The following compute the penalty of the Lipschitz constant
        penalty_coefficient = 10.
        # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
        # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
        f_gradient = torch.autograd.grad(torch.sum(inter_f), interpolated,
                                         create_graph=True, retain_graph=True)[0]
        f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
        f_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - 1.0) ** 2)
        domain_losses = torch.mean(weights * pred) + f_gradient_penalty

        loss = torch.mean(train_losses) + self.mu * domain_losses
        if self.dial_weight is not None:
            return loss + self.dial_weight * target_loss, alpha
        else:
            return loss, alpha

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class MdmnMLP(MdmnBase):

    def __init__(self, configs):
        """
        MDMN with MLP
        """

        super().__init__(configs)

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}

        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])

        self.domain_net = nn.Linear(configs["hidden_layers"][-1],
                                    self.num_domains)


class MdmnConv(MdmnBase):

    def __init__(self, configs):
        """
        MDMN with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.class_net = MLPNet(cls_configs)

        dom_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["dom_fc_layers"],
                       "output_dim": self.num_domains,
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.domain_net = MLPNet(dom_configs)


# MSDA
class MsdaBase(nn.Module):

    def __init__(self):
        """
        Moment matching for multi-source domain adaptation. ICCV 2019
        """

        super().__init__()
        self.num_src_domains = None
        self.feature_net = None
        self.class_net1 = None
        self.class_net2 = None
        self.mu = None
        self.dial_weight = None

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(training loss, discrepancy loss)
        """

        batch_size = tinputs.shape[0]
        if len(tinputs.shape) == 2: #features
            img_shape = [tinputs.shape[-1]]
        else: #img
            img_shape = tinputs.shape[-3:]

        # Compute features
        input_features = torch.stack((tinputs, *sinputs), dim=1).view((batch_size*(self.num_src_domains+1), *img_shape))
        features = self.feature_net(input_features)
        st_features = features.view(batch_size, self.num_src_domains+1, -1)
        t_features = st_features[:, 0, ...]
        s_features = [st_features[:, x+1, ...] for x in range(self.num_src_domains)]

        # Loss and predictions
        input_classifier = features
        
        preds1 = self.class_net1(input_classifier).view((batch_size, self.num_src_domains+1, -1))
        t_pred1 = F.softmax(preds1[:, 0, ...], dim=1)
        logprobs1 = [F.log_softmax(preds1[:, i, ...], dim=1) for i in range(1, self.num_src_domains+1)]
        train_loss1 = sum([torch.mean(F.nll_loss(logprobs1[i], soutputs[i])) for i in range(self.num_src_domains)])

        preds2 = self.class_net2(input_classifier).view((batch_size, self.num_src_domains+1, -1))
        t_pred2 = F.softmax(preds2[:, 0, ...], dim=1)
        logprobs2 = [F.log_softmax(preds2[:, i, ...], dim=1) for i in range(1, self.num_src_domains+1)]
        train_loss2 = sum([torch.mean(F.nll_loss(logprobs2[i], soutputs[i])) for i in range(self.num_src_domains)])

        # Combined by MSDA
        # https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/be3aef641719c0020c8bf11d8ed8b9df79736f6a/M3SDA/code_MSDA_digit/solver_MSDA.py#L286
        loss_msda = utils.msda_regulizer(s_features, t_features)
            
        train_loss = train_loss1 + train_loss2 + (self.mu * loss_msda)
        disc_loss = torch.mean(torch.abs(t_pred1 - t_pred2))
        
        if self.dial_weight is not None:
            dial_loss = train_loss + (self.dial_weight * utils.entropy(t_pred1))
            return train_loss, disc_loss, dial_loss
        else:
            return train_loss, disc_loss, 0.0

        return 

    def inference(self, x):

        x = self.feature_net(x)
        # Classification probability.
        return F.log_softmax(self.class_net1(x), dim=1)


class MsdaMLP(MsdaBase):

    def __init__(self, configs):
        """
        MSDA with MLP
        """

        super().__init__()
        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}
        self.feature_net = MLPNet(fea_configs)
        self.G_params = self.feature_net.parameters()

        self.num_src_domains = configs["num_src_domains"]
        # Parameter of the final softmax classification layer.
        self.class_net1 = nn.Linear(configs["hidden_layers"][-1],
                                    configs["num_classes"])
        self.class_net2 = nn.Linear(configs["hidden_layers"][-1],
                                    configs["num_classes"])
        self.C1_params = self.class_net1.parameters()
        self.C2_params = self.class_net2.parameters()

        self.mu = 5e-4
        self.dial_weight = configs['dial_weight']


class MsdaConv(MsdaBase):

    def __init__(self, configs):
        """
        MSDA with convolution feature extractor
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"][:-1],
                       "output_dim": configs["cls_fc_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}

        self.feature_net = nn.Sequential(ConvNet(configs), MLPNet(cls_configs))
        self.G_params = self.feature_net.parameters()

        self.class_net1 = nn.Linear(configs["cls_fc_layers"][-1], configs["num_classes"])
        self.class_net2 = nn.Linear(configs["cls_fc_layers"][-1], configs["num_classes"])

        self.C1_params = self.class_net1.parameters()
        self.C2_params = self.class_net2.parameters()

        self.dial_weight = configs['dial_weight']
        self.mu = 1e-7

#DIAL
class MSDialBase(nn.Module):

    def __init__(self, configs):
        """
        MS-DIAL model
        """

        super().__init__()

        self.num_src_domains = configs["num_src_domains"]
        self.num_domains = configs["num_src_domains"] + 1
        self.dial_weight = configs['dial_weight']

    def forward(self, sinputs, soutputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(loss, domain weights)
        """

        batch_size = tinputs.shape[0]
        if len(tinputs.shape) == 2: #features
            img_shape = [tinputs.shape[-1]]
        else: #img
            img_shape = tinputs.shape[-3:]

        input_features = torch.stack((tinputs, *sinputs), dim=1).view(-1, *img_shape)
        
        features = self.feature_net(input_features)
        logits = self.class_net(features)
        
        st_logits = logits.view((batch_size, self.num_domains, -1))
        t_logits = st_logits[:, 0, ...]
        s_logits = st_logits[:, 1:, ...]

        s_logits = s_logits.transpose(0, 1)
        s_logits = torch.reshape(s_logits, (batch_size*self.num_src_domains, -1))
        s_target = torch.cat(soutputs, dim=0)
        train_losses = F.cross_entropy(s_logits, s_target)
        target_loss  = utils.entropy(t_logits)

        loss = train_losses + self.dial_weight * target_loss

        return loss, None

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class MSDialMLP(MSDialBase):

    def __init__(self, configs):
        """
        MDMN with MLP
        """

        super().__init__(configs)

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": True}

        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])


class MSDialConv(MSDialBase):

    def __init__(self, configs):
        """
        MS-DIAL with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["cls_fc_layers"],
                       "output_dim": configs["num_classes"],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
                       
        self.class_net = MLPNet(cls_configs)
