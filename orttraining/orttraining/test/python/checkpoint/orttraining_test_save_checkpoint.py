#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pickle
import argparse
from itertools import islice

import torch
import torch.distributed as dist

from onnxruntime import set_seed
from onnxruntime.training import amp, optim, orttrainer
from _test_helpers import distributed_setup, _load_pytorch_transformer_model, train, save, chunkify

def single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir)

def single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_data, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir)

@distributed_setup
def data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    if world_rank == 0:
        save(trainer, checkpoint_dir)

@distributed_setup
def data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    if world_rank == 0:
        save(trainer, checkpoint_dir)

@distributed_setup
def distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir, 'state_dict_'+str(world_rank))

@distributed_setup
def distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model as example
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    optim_config = optim.LambConfig(lr=learning_rate)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    train_chunk = next(islice(chunkify(train_data, world_size), world_rank, None))
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)

    # run train steps
    train(trainer, train_chunk, batcher_fn)

    # save current model parameters as a checkpoint
    save(trainer, checkpoint_dir, 'state_dict_'+str(world_rank))

function_map = {
    'single_node_full_precision': single_node_full_precision,
    'single_node_mixed_precision': single_node_mixed_precision,
    'data_parallelism_full_precision': data_parallelism_full_precision,
    'data_parallelism_mixed_precision': data_parallelism_mixed_precision,
    'distributed_zero_full_precision': distributed_zero_full_precision,
    'distributed_zero_mixed_precision': distributed_zero_mixed_precision
}
parser = argparse.ArgumentParser(description='Save states of trainers')
parser.add_argument('--scenario', choices=function_map.keys(), help='training scenario to save states', required=True)
parser.add_argument('--checkpoint_dir', help='path to the directory where checkpoints can be saved', required=True)
args = parser.parse_args()
function_map[args.scenario](checkpoint_dir = args.checkpoint_dir)
