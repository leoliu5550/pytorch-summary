import typing
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils._pytree import tree_map



def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None,pr = True):
    result = summary_string(
        model, input_size, batch_size, device, dtypes)
    if pr:
        print(result["summary_str"])
    return result["total_params"],result["trainable_params"]


def tensor_size(tensor: typing.Any) -> typing.Optional[typing.List[int]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    return list(tensor.size())


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)
    summary_str = ''
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            input_shape = tree_map(tensor_size, input)
            if len(input_shape) == 1:  # backwards compatibility
                input_shape = input_shape[0]
            summary[m_key]["input_shape"] = input_shape
            summary[m_key]["output_shape"] = tree_map(tensor_size, input)

            params = 0
            summary[m_key]["trainable"] = False
            for p in module.parameters(recurse=False):
                params += np.prod(list(p.size()))
                summary[m_key]["trainable"] |= p.requires_grad
            summary[m_key]["nb_params"] = params

        if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)]
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(*x)
    # remove these hooks
    for h in hooks:
        h.remove()
    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"
    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    # return summary_str, (total_params, trainable_params)
    return {
        "summary_str":summary_str,
        "total_params" : total_params,
        "trainable_params" : trainable_params,
        "non_trainable_params":(total_params - trainable_params),
        "input_size" : total_input_size,
        "forward_backward_size": total_output_size,
        "total_params_size":total_params_size,
        "total_size":total_size
    }