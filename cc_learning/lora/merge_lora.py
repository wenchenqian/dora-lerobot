import argparse
import torch
import pangu
import os

from megatron.core import mpu
from megatron.training import initialize_megatron, get_args, get_tokenizer, get_timers, print_rank_0
from megatron.training.training import get_model, load_checkpoint, unwrap_model
from megatron.inference.text_generation import generate
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.training.checkpointing import save_checkpoint

from pangu.training.arguments import _add_pangu_args
from pretrain_gpt import model_provider, train_valid_test_datasets_provider, forward_step
from tools.process_data.preprocess_data_sft import add_data_args, add_output_args
from inference_gpt import add_extra_generate_args
from pangu.tasks.finetune.lora.utils import is_enable_lora
from peft import PeftModel, LoraModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--output-prefix', type=str,
                       help='Path to binary output file without suffix')


def add_extra_args_algo(parser):
    parser = add_extra_generate_args(parser)
    _add_pangu_args(parser)
    add_data_args(parser)
    add_tokenizer_args(parser)
    return parser


def main():
    initialize_megatron(extra_args_provider=add_extra_args_algo)

    args = get_args()

    pangu_model = get_model(model_provider, wrap_with_ddp=True)
    unwrapped_model = unwrap_model(pangu_model)

    if args.load is not None:
        torch.distributed.barrier()
        if not os.path.exists(args.load) \
                or not os.path.exists(f"{args.load}/latest_checkpointed_iteration.txt"):
            raise FileNotFoundError(
                f"args.load {args.load} is not valid (may be miss 'latest_checkpointed_iteration.txt' file )")
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(pangu_model, None, None)
        torch.distributed.barrier()
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    if isinstance(pangu_model, list):
        assert len(pangu_model) == 1, "non-pipeline-parallel schedule does not support model chunking"

    if is_enable_lora():
        for model_item in pangu_model:
            while not isinstance(model_item, PeftModel):
                model_item = model_item.module
            model_item.merge_and_unload()
        print('lora merge complete!')

    save_checkpoint(1, pangu_model, None, None, 0)


if __name__ == '__main__':
    main()
