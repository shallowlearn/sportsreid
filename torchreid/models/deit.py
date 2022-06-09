'''
We use models pretrained on ImageNet from https://github.com/facebookresearch/deit/blob/main/README_deit.md
and from https://github.com/facebookresearch/deit
'''

from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .models_v2 import deit_huge_patch14_LS, deit_large_patch16_LS, deit_base_patch16_LS

__all__ = ['deit_b_16', 'deit_t_16', 'deit_s_16', 'deit_bd_16', 'deit_td_16', 'deit_sd_16',
           'deit_bd_16_384', 'deit_h_14_ls', 'deit_l_16_ls', 'deit_b_16_ls']

pretrained_urls = {
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

##########
# Network architecture
##########
class DeiT(nn.Module):
    """DeiT Network.
    """

    def __init__(
        self,
        num_classes,
        name,
        pretrained=True,
        feature_dim=768,
        loss='softmax',
        pretrained_21k=True,
        img_size=None,
        **kwargs
    ):
        super().__init__()

        # Create a pretrained deit model
        if name == 'deit_b_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        if name == 'deit_bd_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
        if name == 'deit_s_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        if name == 'deit_sd_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224', pretrained=True)
        if name == 'deit_t_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        if name == 'deit_td_16':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        if name == 'deit_bd_16_384':
            model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        if name == 'deit_h_14_ls':
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
            assert img_size == 224
            model = deit_huge_patch14_LS(pretrained=True, pretrained_21k=pretrained_21k, img_size=img_size)
        if name == 'deit_l_16_ls':
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
            assert img_size == 224 or img_size == 384
            model = deit_large_patch16_LS(pretrained=True, pretrained_21k=pretrained_21k, img_size=img_size)
        if name == 'deit_b_16_ls':
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
            assert img_size == 224 or img_size == 384
            model = deit_base_patch16_LS(pretrained=True, pretrained_21k=pretrained_21k, img_size=img_size)

        self.loss = loss
        self.feature_dim = model.head.weight.shape[1]

        # Replace the classification layer in the model with an Identity layer
        model.head = Identity()
        # Distilled model has two heads
        if '_td_' in name or '_bd_' in name or '_sd_' in name:
            model.head_dist = Identity()
            print("Using distilled model")
            self.distilled = True
        else:
            self.distilled = False

        self.model = model

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # # fully connected layer
        self.fc = self._construct_fc_layer(
            int(min(512, self.feature_dim)), self.feature_dim, dropout_p=None
        )
        # self.fc = None

        # Add our own classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # self._init_params()

    # def _make_layer(
    #     self,
    #     block,
    #     layer,
    #     in_channels,
    #     out_channels,
    #     reduce_spatial_size,
    #     IN=False
    # ):
    #     layers = []
    #
    #     layers.append(block(in_channels, out_channels, IN=IN))
    #     for i in range(1, layer):
    #         layers.append(block(out_channels, out_channels, IN=IN))
    #
    #     if reduce_spatial_size:
    #         layers.append(
    #             nn.Sequential(
    #                 Conv1x1(out_channels, out_channels),
    #                 nn.AvgPool2d(2, stride=2)
    #             )
    #         )
    #
    #     return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            #layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    # def _init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(
    #                 m.weight, mode='fan_out', nonlinearity='relu'
    #             )
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        if not self.distilled:
            x = self.model(x)
        else:
            # In case of distilled, there will be two feature vectors.
            # We just take average for now
            x, y = self.forward_features(x)
            x = (x + y)/2
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        # v = self.global_avgpool(x)
        v = x
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

#
# def init_pretrained_weights(model, key=''):
#     """Initializes model with pretrained weights.
#
#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     import os
#     import errno
#     import gdown
#     from collections import OrderedDict
#
#     def _get_torch_home():
#         ENV_TORCH_HOME = 'TORCH_HOME'
#         ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
#         DEFAULT_CACHE_DIR = '~/.cache'
#         torch_home = os.path.expanduser(
#             os.getenv(
#                 ENV_TORCH_HOME,
#                 os.path.join(
#                     os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
#                 )
#             )
#         )
#         return torch_home
#
#     torch_home = _get_torch_home()
#     model_dir = os.path.join(torch_home, 'checkpoints')
#     try:
#         os.makedirs(model_dir)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             # Directory already exists, ignore.
#             pass
#         else:
#             # Unexpected OSError, re-raise.
#             raise
#     filename = key + '_imagenet.pth'
#     cached_file = os.path.join(model_dir, filename)
#
#     if not os.path.exists(cached_file):
#         gdown.download(pretrained_urls[key], cached_file, quiet=False)
#
#     state_dict = torch.load(cached_file)
#     model_dict = model.state_dict()
#     new_state_dict = OrderedDict()
#     matched_layers, discarded_layers = [], []
#
#     for k, v in state_dict.items():
#         if k.startswith('module.'):
#             k = k[7:] # discard module.
#
#         if k in model_dict and model_dict[k].size() == v.size():
#             new_state_dict[k] = v
#             matched_layers.append(k)
#         else:
#             discarded_layers.append(k)
#
#     model_dict.update(new_state_dict)
#     model.load_state_dict(model_dict)
#
#     if len(matched_layers) == 0:
#         warnings.warn(
#             'The pretrained weights from "{}" cannot be loaded, '
#             'please check the key names manually '
#             '(** ignored and continue **)'.format(cached_file)
#         )
#     else:
#         print(
#             'Successfully loaded imagenet pretrained weights from "{}"'.
#             format(cached_file)
#         )
#         if len(discarded_layers) > 0:
#             print(
#                 '** The following layers are discarded '
#                 'due to unmatched keys or layer size: {}'.
#                 format(discarded_layers)
#             )


##########
# Instantiation
##########

def deit_b_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_b_16',
        pretrained=pretrained,
        feature_dim=768,
        loss=loss,
        **kwargs
    )
    return model

def deit_bd_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_bd_16',
        pretrained=pretrained,
        feature_dim=768,
        loss=loss,
        **kwargs
    )
    return model

def deit_bd_16_384(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_bd_16_384',
        pretrained=pretrained,
        feature_dim=768,
        loss=loss,
        **kwargs
    )
    return model

def deit_s_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_s_16',
        pretrained=pretrained,
        feature_dim=384,
        loss=loss,
        **kwargs
    )
    return model

def deit_sd_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_sd_16',
        pretrained=pretrained,
        feature_dim=384,
        loss=loss,
        **kwargs
    )
    return model

def deit_t_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_t_16',
        pretrained=pretrained,
        feature_dim=192,
        loss=loss,
        **kwargs
    )
    return model

def deit_h_14_ls(num_classes=1000, pretrained=True, loss='softmax', pretrained_21k=True, img_size=None, **kwargs):
    model = DeiT(
        num_classes,
        name='deit_h_14_ls',
        pretrained=pretrained,
        feature_dim=192,
        loss=loss,
        pretrained_21k=pretrained_21k,
        img_size=img_size,
        **kwargs
    )
    return model

def deit_l_16_ls(num_classes=1000, pretrained=True, loss='softmax', pretrained_21k=True, img_size=None, **kwargs):
    model = DeiT(
        num_classes,
        name='deit_l_16_ls',
        pretrained=pretrained,
        feature_dim=192,
        loss=loss,
        pretrained_21k=pretrained_21k,
        img_size=img_size,
        **kwargs
    )
    return model

def deit_b_16_ls(num_classes=1000, pretrained=True, loss='softmax', pretrained_21k=True, img_size=None, **kwargs):
    model = DeiT(
        num_classes,
        name='deit_b_16_ls',
        pretrained=pretrained,
        feature_dim=192,
        loss=loss,
        pretrained_21k=pretrained_21k,
        img_size=img_size,
        **kwargs
    )
    return model

def deit_td_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = DeiT(
        num_classes,
        name='deit_td_16',
        pretrained=pretrained,
        feature_dim=192,
        loss=loss,
        **kwargs
    )
    return model



