from collections import OrderedDict

import torch


def load_weights_from_checkpoint(model, path):
    ckpt = torch.load(path + '.ckpt', map_location=lambda storage, loc: storage)
    state_dict = ckpt["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("network.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
