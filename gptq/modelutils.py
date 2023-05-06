import torch
import torch.nn as nn

from . import quant_v1
from . import quant_v2

GPTQVERSION = 1


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def set_gptq_version(version):
    global GPTQVERSION
    GPTQVERSION = version


def make_quant(*args, **kwargs):
    if GPTQVERSION == 0:
        return quant_v1.make_quant(*args, **kwargs)

    if GPTQVERSION == 1:
        return quant_v2.make_quant(*args, **kwargs)
