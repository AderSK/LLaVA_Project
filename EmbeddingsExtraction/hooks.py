from pathlib import Path
import numpy as np
import os
import torch.nn as nn

def hook_backbone(activation_backbone, layer: str):
    """
    layer_nr: string name of layer
    Example usage:
    Register hook for the 12th layer (where groot extract all the data)
    groot_policy.model.backbone.layers[12].register_forward_hook(hook_backbone(activation_backbone, layer))
    """
    def hook(model, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        # if activation_backbone.get(layer, None):
        #     activation_backbone[layer].append(activation)
        # else:
        activation_backbone[layer] = [activation]
        return output
    return hook

def register_hook_backbone(model, layers=[11]):
    """
    Registers the hook for layers defined in the 'layers' list of the backbone at once
    Arguments:
        model: groot_policy.model.backbone
        layers: list of layer numbers to extract the activations from
    Returns:
        activation_backbone: dictionary containing features from the backbone (Eagle 2)
    """
    # Define hooks to extract features from the backbone (Eagle 2)
    activation_backbone = {}

    for layer in layers:
        layer_name = str(layer)
        model.layers[layer].register_forward_hook(hook_backbone(activation_backbone, layer_name))

    return activation_backbone


def hook_encoder_header(activation_action_head, name):
    """
    name: name of activation head
    Example usage:
    Register hooks for the action heads of the Diffusion Transformer
    groot_policy.model.action_head.layer.register_forward_hook(hook_backbone(activation_backbone, layer_nr))
    """
    def hook(model, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        if name not in activation_action_head:
            activation_action_head[name] = []
        activation_action_head[name].append(activation)
        return output
    return hook


def register_hook_encoder_header(model):
    """
    Register the hook for all heads of the Diffusion Transformer at once
    Arguments:
    model: groot_policy.model.action_head
    Returns:
    activation_action_head: Define hooks dictionary to extract activations
    """
    activation_action_head = {}
    # Register hook for all layers action_head
    for i, layer in enumerate(model.transformer_blocks):
        name = f"self_attention_{i}" if i%2==0 else f"cross_attention_{i}"
        layer.register_forward_hook(hook_encoder_header(activation_action_head, name))
    return activation_action_head
# Clean up hooks
def clear_all_hooks(model):
    """Clear all forward hooks from a PyTorch model recursively.
        Arguments:
         model: Gr00tPolicy
    """

    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()

# a simple example of a hook writing activations into a layer-labeled dictionary
#hooked_res = {"hidden_states": None}
#def forward_hook(model, input, output):
#    """
#    Example usage:
#    hook_gen=model.language_model.model.layers[24].register_forward_hook(forward_hook)
#    """
#    if hooked_res["hidden_states"] is not None:
#        hooked_res["hidden_states"].append(output)
#    else:
#        hooked_res["hidden_states"] = [output]
#    return output

