# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Any, List, Optional, Union
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.init as init

from peft.tuners.lora.layer import LoraLayer
from peft.utils.other import transpose
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core import parallel_state
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.module import MegatronModule
from mindspeed.ops.gmm import npu_gmm
from pangu.core.utils import checkpoint_non_reentrant, print_saved_tensors
from pangu.core.transformer.moe.experts import Experts


class LoraParallelLinearMoE(nn.Module, LoraLayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            backend,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self.backend = backend
        self.is_parallel_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.is_expert = base_layer.is_expert

        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method
        input_is_parallel = True
        gather_output = False
        if isinstance(base_layer, self.backend.RowParallelLinear):
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output,
            **parallel_linear_kwargs,
        )

        self.is_target_conv_1d_layer = False

    def update_layer(
            self,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora=False,
            init_method=init.xavier_normal_,
            input_is_parallel=True,
            gather_output=False,
            **parallel_linear_kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        megatron_config = parallel_linear_kwargs["megatron_config"]
        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        megatron_config.params_dtype = torch.float32
        if self.is_parallel_a:
            lora_a = self.backend.RowParallelLinear(
                input_size=self.in_features,
                output_size=r,
                bias=False,
                input_is_parallel=input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
                config=megatron_config,
                is_expert=self.is_expert,
            )
            lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
        else:
            lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            lora_b = self.backend.ColumnParallelLinear(
                input_size=r,
                output_size=self.out_features,
                bias=False,
                gather_output=gather_output,
                init_method=init_method,
                config=megatron_config,
                is_expert=self.is_expert,
            )

        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / (r ** 0.5)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                dropout_x = dropout(x)
                lora_result = lora_A(dropout_x)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_B(lora_result)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling
                result = result + lora_result

            result = result.to(previous_dtype)

        return result, bias


class LoraGroupGemmExperts(MegatronModule, LoraLayer):
    # Lora implemented for GroupedMLP
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_A2", "lora_B", "lora_B2", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "lora_dropout2")

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = True,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(base_layer.config)
        setattr(base_layer, 'input_size', base_layer.config.hidden_size)
        setattr(base_layer, 'output_size', base_layer.config.moe_routed_expert_hidden_size)
        LoraLayer.__init__(self, base_layer, **kwargs)

        if self.base_layer.config.moe_extended_tp:
            tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        elif self.base_layer.config.moe_ep_over_sp:
            tp_size = 1
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.out_features = divide(self.out_features, tp_size)

        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.lora_A2 = nn.ModuleDict({})
        self.lora_B2 = nn.ModuleDict({})
        self.lora_dropout2 = nn.ModuleDict({})
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.cur_adapter = None

        self.expert_parallel = self.base_layer.expert_parallel
        for name, param in self.named_parameters():
            setattr(param, "allreduce", not self.expert_parallel)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
            lora_dropout_layer2 = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
            lora_dropout_layer2 = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_dropout2.update(nn.ModuleDict({adapter_name: lora_dropout_layer2}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[adapter_name] = nn.Linear(self.base_layer.num_local_experts * self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(self.base_layer.num_local_experts * r, self.out_features * 2,
                                                  bias=False)
            self.lora_A2[adapter_name] = nn.Linear(self.base_layer.num_local_experts * self.out_features, r, bias=False)
            self.lora_B2[adapter_name] = nn.Linear(self.base_layer.num_local_experts * r, self.in_features, bias=False)
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.get_base_layer(), "weight1", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_A2.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A2[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A2[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B2[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    delta1, delta2 = self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight1.data.clone()
                    orig_weights += delta1.view(orig_weights.shape)
                    orig_weights2 = base_layer.weight2.data.clone()
                    orig_weights2 += delta2.view(orig_weights2.shape)

                    if not torch.isfinite(orig_weights).all() or not torch.isfinite(orig_weights2).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight1.data = orig_weights
                    base_layer.weight2.data = orig_weights2
                else:
                    delta1, delta2 = self.get_delta_weight(active_adapter)
                    base_layer.weight1.data += delta1.view(base_layer.weight1.shape)
                    base_layer.weight2.data += delta2.view(base_layer.weight2.shape)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                delta1, delta2 = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight1.data -= delta1.view(self.get_base_layer().weight1.shape)
                self.get_base_layer().weight2.data -= delta2.view(self.get_base_layer().weight2.shape)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        weight_A2 = self.lora_A2[adapter].weight
        weight_B2 = self.lora_B2[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            weight_A2 = weight_A2.float()
            weight_B2 = weight_B2.float()

        output_tensor = torch.bmm(
            weight_A.view(self.base_layer.num_local_experts, self.base_layer.config.hidden_size, -1),
            weight_B.view(self.base_layer.num_local_experts, -1, self.out_features * 2)) * self.scaling[adapter]
        output_tensor2 = torch.bmm(weight_A2.view(self.base_layer.num_local_experts, self.out_features, -1),
                                   weight_B2.view(self.base_layer.num_local_experts, -1,
                                                  self.base_layer.config.hidden_size)) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            output_tensor2 = output_tensor2.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_A2[adapter].weight.data = weight_A2.to(dtype)
            self.lora_B2[adapter].weight.data = weight_B2.to(dtype)

        return output_tensor, output_tensor2

    def forward(
            self,
            hidden_states: torch.Tensor,
            probs: torch.Tensor,
            indices: torch.Tensor,
            faked: bool = False,
    ) -> Optional[torch.Tensor]:

        assert not self.base_layer.config.moe_permute_recompute and not self.base_layer.config.moe_activation_recompute

        previous_dtype = hidden_states.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(hidden_states, probs, indices, faked)
        elif self.merged:
            return self.base_layer(hidden_states, probs, indices, faked)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                self.cur_adapter = active_adapter
                hidden_states = hidden_states.to(self.lora_A[active_adapter].weight.dtype)
                # result += lora_B(lora_A(dropout(permuted_local_hidden_states))) * scaling

                with print_saved_tensors("experts::permute_up", self.base_layer.config):
                    expert_hiddens, permuted_probs = self.lora_up(hidden_states, probs, indices)

                with print_saved_tensors("experts::down", self.base_layer.config):
                    expert_outputs = self.lora_down(expert_hiddens, permuted_probs)

                event = torch.cuda.default_stream().record_event()
                with print_saved_tensors("experts::unpermute", self.base_layer.config):
                    routed_output = self.base_layer.token_dispatcher.token_unpermutation(expert_outputs)

                if self.base_layer.config.moe_shared_expert_hidden_size is not None:
                    with print_saved_tensors("experts::shared", self.base_layer.config):
                        Experts.shared_expert_stream.wait_event(event)
                        with torch.cuda.stream(Experts.shared_expert_stream):
                            shared_output, shared_bias = self.base_layer.shared_expert(hidden_states)
                            if shared_bias is not None:
                                shared_output = shared_output + shared_bias
                                hidden_states.record_stream(Experts.shared_expert_stream)
                        torch.cuda.default_stream().wait_stream(Experts.shared_expert_stream)
                        output = routed_output + shared_output
                        shared_output.record_stream(torch.cuda.default_stream())
                else:
                    output = routed_output

                return output

    def lora_up(self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor):
        permuted, tokens_per_expert, permuted_probs, _ = self.base_layer.token_dispatcher.token_permutation(
            hidden_states, probs, indices
        )

        self.tokens_per_expert = tokens_per_expert.to(device=permuted.device)
        self.group_list = self.tokens_per_expert.cumsum(dim=0).tolist()

        if permuted.nelement() != 0:
            fc1_output = npu_gmm(
                permuted, self.base_layer.weight1, bias=None, group_list=self.group_list, group_type=0
            )
            if self.base_layer.add_bias:
                b1 = self.base_layer.bias1.view(self.base_layer.num_local_experts, 1, -1)
                fc1_output = fc1_output + torch.repeat_interleave(b1, self.tokens_per_expert, dim=0)

            lora_a = npu_gmm(permuted, self.lora_A[self.cur_adapter].weight.view(self.base_layer.num_local_experts,
                                                                                 self.base_layer.config.hidden_size,
                                                                                 -1),
                             bias=None, group_list=self.group_list, group_type=0)
            lora_a = self.lora_dropout[self.cur_adapter](lora_a)
            lora_b = npu_gmm(lora_a, self.lora_B[self.cur_adapter].weight.view(self.base_layer.num_local_experts, -1,
                                                                               self.out_features * 2),
                             bias=None, group_list=self.group_list, group_type=0) * self.scaling[self.cur_adapter]
            fc1_output = fc1_output + lora_b
        else:
            w1 = self.base_layer.weight1.view(self.base_layer.config.hidden_size, -1)
            fc1_output = torch.matmul(permuted, w1)
            lora_a = torch.matmul(permuted,
                                  self.lora_A[self.cur_adapter].weight.view(self.base_layer.config.hidden_size, -1))
            lora_b = torch.matmul(lora_a, self.lora_B[self.cur_adapter].weight.view(-1, self.out_features * 2)).reshape(
                fc1_output.shape)
            fc1_output = fc1_output + lora_b

        return fc1_output, permuted_probs

    def lora_down(self, up_hidden_states: torch.Tensor, permuted_probs):
        if permuted_probs is not None:
            up_hidden_states = (self.base_layer.activation_func(up_hidden_states) *
                                permuted_probs.reshape(*up_hidden_states.shape[:-1], 1))
        else:
            up_hidden_states = self.base_layer.activation_func(up_hidden_states)
        if up_hidden_states.nelement() != 0:
            fc2_output = npu_gmm(
                up_hidden_states, self.base_layer.weight2, bias=None, group_list=self.group_list, group_type=0
            )

            if self.base_layer.add_bias:
                b2 = self.base_layer.bias2.view(self.base_layer.num_local_experts, 1, -1)
                fc2_output = fc2_output + torch.repeat_interleave(
                    b2 / parallel_state.get_tensor_model_parallel_world_size(),
                    self.tokens_per_expert,
                    dim=0,
                )

            lora_a = npu_gmm(up_hidden_states,
                             self.lora_A2[self.cur_adapter].weight.view(self.base_layer.num_local_experts,
                                                                        self.out_features, -1),
                             bias=None, group_list=self.group_list, group_type=0)
            lora_a = self.lora_dropout2[self.cur_adapter](lora_a)
            lora_b = npu_gmm(lora_a, self.lora_B2[self.cur_adapter].weight.view(self.base_layer.num_local_experts, -1,
                                                                                self.base_layer.config.hidden_size),
                             bias=None, group_list=self.group_list, group_type=0) * self.scaling[self.cur_adapter]
            fc2_output = fc2_output + lora_b
        else:
            w2 = self.base_layer.weight2.view(-1, self.base_layer.config.hidden_size)
            fc2_output = torch.matmul(up_hidden_states, w2)
            lora_a = torch.matmul(up_hidden_states, self.lora_A2[self.cur_adapter].weight.view(
                self.base_layer.num_local_experts * self.out_features, -1))
            lora_b = torch.matmul(lora_a, self.lora_B2[self.cur_adapter].weight.view(-1,
                                                                                     self.base_layer.num_local_experts * self.base_layer.config.hidden_size)).reshape(
                fc2_output.shape)
            fc2_output = fc2_output + lora_b

        return fc2_output

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep