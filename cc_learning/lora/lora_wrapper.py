from functools import wraps

import torch
import megatron
from megatron.training import get_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType

from pangu.tasks.finetune.lora.utils import is_enable_lora
from pangu.compression.lora_moe import LoraParallelLinearMoE

import peft
from packaging import version
from peft import LoraConfig, get_peft_model, PeftModel, LoraModel


def unwrap_model_wrapper(unwrap_func):
    @wraps(unwrap_func)
    def wrapper(*args, **kwargs):
        model = unwrap_func(*args, **kwargs)

        return_list = True
        if not isinstance(model, list):
            model = [model]
            return_list = False
        unwrapped_model = []
        for model_module in model:
            if isinstance(model_module, PeftModel):
                model_module = model_module.base_model
            if isinstance(model_module, LoraModel):
                model_module = model_module.model
            unwrapped_model.append(model_module)
        if not return_list:
            return unwrapped_model[0]
        return unwrapped_model

    return wrapper


def model_provider_func_wrapper(model_provider_func):
    @wraps(model_provider_func)
    def wrapper(*args, **kwargs):
        model = model_provider_func(*args, **kwargs)
        args = get_args()

        if is_enable_lora():
            if version.parse(peft.__version__) <= version.parse('0.11.1'):
                setattr(peft.tuners.lora.LoraLayer, 'merge', peft.tuners.lora.Linear.merge)
                setattr(peft.tuners.lora.LoraLayer, 'unmerge', peft.tuners.lora.Linear.unmerge)
                setattr(peft.tuners.lora.LoraLayer, 'get_delta_weight', peft.tuners.lora.Linear.get_delta_weight)
            from peft.tuners.lora import tp_layer
            tp_layer.LoraParallelLinear = LoraParallelLinearMoE

            config = core_transformer_config_from_args(args)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                megatron_config=config,
                megatron_core="megatron.core",
            )

            # model = get_peft_model(model, lora_config)
            # model.add_module('module', model.get_base_model())

            def _hook(_module, _x_in, _x_out):
                """ Extract the feature map of model"""
                _x_out.requires_grad_(True)

            def _create_hooks(_model, layer):
                """ Make the hooks function"""
                for name, module in _model.named_modules():
                    if isinstance(module, megatron.core.tensor_parallel.layers.VocabParallelEmbedding):
                        _name = name.split('.')[-1]
                        if _name in layer:
                            module.register_forward_hook(_hook)

            model = get_peft_model(model, lora_config)
            # import pdb
            # pdb.set_trace()
            model.print_trainable_parameters()

            if args.recompute_method == 'block' and args.recompute_granularity == 'full':
                _create_hooks(model, args.lora_register_forward_hook)

            for module in model.modules():
                # LoRA Linear Layer need all reduce
                if isinstance(module, torch.nn.Linear):
                    setattr(module.weight, 'sequence_parallel', config.sequence_parallel)
                # Other layers if is frozen, do not need all reduce.
                for param in module.parameters():
                    if not param.requires_grad and hasattr(param, 'sequence_parallel'):
                        delattr(param, 'sequence_parallel')

            megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES = tuple(
                list(megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES) + [PeftModel, LoraModel]
            )

        return model

    return wrapper


def get_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
        model_provider_func = model_provider_func_wrapper(model_provider_func)
        model = fn(model_provider_func, model_type, wrap_with_ddp)
        if is_enable_lora():
            for model_module in model:
                model_module.broadcast_params()
        return model

    return wrapper