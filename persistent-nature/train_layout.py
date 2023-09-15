# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from external.stylegan.training import training_loop
from external.stylegan.metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        # add a temporary resolution for kwargs, will be overridden later
        dataset_kwargs = dnnlib.EasyDict(
            class_name='external.stylegan.training.dataset.ImageFolderDataset',
            path=data, resolution=64, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

## ADDED: additional training options
# which part of the model is being trained
@click.option('--training-mode', help='which part of the model is being trained', type=str, required=True)

# dataset options
@click.option('--pose', help='pose path', metavar='[ZIP|DIR]', type=str)
@click.option('--img-resolution', help='image resolution', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--depth-scale', help='depth scale factor', metavar='FLOAT', type=click.FloatRange(min=0), default=16, show_default=True)
@click.option('--depth-clip', help='depth clip factor', metavar='FLOAT', type=click.FloatRange(min=0), default=20, show_default=True)
@click.option('--use-disp', help='train using disparity', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--fov-mean', help='mean fov', metavar='FLOAT', type=click.FloatRange(min=0,max=180), default=60, show_default=True)
@click.option('--fov-std', help='std fov', metavar='FLOAT', type=click.FloatRange(min=0,max=180), default=0, show_default=True)

# layout generator options
@click.option('--z-dim', help='mapping layers input dimension', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--num-layers', help='number of generator layers', type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--num-decoder-ch', help='number of layout features', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--voxel-res', help='voxel res of decoder', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--voxel-size', help='voxel size', metavar='FLOAT', type=click.FloatRange(min=0), default=0.15, show_default=True)

# layout decoder options
@click.option('--feature-nerf', help='does not apply activation to rgb', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--nerf-out-res', help='output resolution of nerf', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--nerf-out-ch', help='output channels of nerf', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--nerf-n-layers', help='layers in nerf mlp', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--nerf-far', help='nerf far bound', metavar='FLOAT', type=click.FloatRange(min=1), default=16., show_default=True)
@click.option('--nerf-samples-per-ray', help='nerf samples per ray', type=click.IntRange(min=1), default=128, show_default=True)

# layout: loss arguments
@click.option('--concat-depth', help='depth to discriminator', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--concat-acc', help='acc to discriminator', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--recon-weight', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=1000., show_default=True)
@click.option('--aug-policy', help='aug policy', type=str, default='translation,color,cutout', required=True)
@click.option('--d-cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--d-cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--ray-lambda-finite-difference', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--ray-lambda-ramp-end', help='ramp function end (kimg)', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--use-wrapped-discriminator', help='wrap discriminator with second acc discriminator', type=bool, default=False, show_default=True)

# upsampler model: architecture
@click.option('--input-resolution', help='upsampler input resolution', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--concat-depth-and-acc', help='upsampler also generates depth, acc', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--use-3d-noise', help='sgan2 3D noise in upsampler', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--layout-model', help='layout model path', type=str)

# upsampler model: losses
@click.option('--lambda-rec', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-up', help='upsample recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-gray-pixel', help='penalty on gray pixels where acc=1', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-gray-pixel-falloff', help='penalty on gray pixels; exponential falloff rate', metavar='FLOAT', type=click.FloatRange(min=0), default=20, show_default=True)
@click.option('--D-ignore-depth-acc', help='ignores depth, acc inputs to discriminator', metavar='BOOL', type=bool, default=True, show_default=True)

# additional arguments for sky model
@click.option('--mask-prob', help='probability to use gt rgb in sky generator output', type=float)


## ADDED: additional training options
# which part of the model is being trained
@click.option('--training-mode', help='which part of the model is being trained', type=str, required=True)

# dataset options
@click.option('--pose', help='pose path', metavar='[ZIP|DIR]', type=str)
@click.option('--img-resolution', help='image resolution', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--depth-scale', help='depth scale factor', metavar='FLOAT', type=click.FloatRange(min=0), default=16, show_default=True)
@click.option('--depth-clip', help='depth clip factor', metavar='FLOAT', type=click.FloatRange(min=0), default=20, show_default=True)
@click.option('--use-disp', help='train using disparity', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--fov-mean', help='mean fov', metavar='FLOAT', type=click.FloatRange(min=0,max=180), default=60, show_default=True)
@click.option('--fov-std', help='std fov', metavar='FLOAT', type=click.FloatRange(min=0,max=180), default=0, show_default=True)

# layout generator options
@click.option('--z-dim', help='mapping layers input dimension', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--num-layers', help='number of generator layers', type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--num-decoder-ch', help='number of layout features', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--voxel-res', help='voxel res of decoder', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--voxel-size', help='voxel size', metavar='FLOAT', type=click.FloatRange(min=0), default=0.15, show_default=True)

# layout decoder options
@click.option('--feature-nerf', help='does not apply activation to rgb', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--nerf-out-res', help='output resolution of nerf', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--nerf-out-ch', help='output channels of nerf', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--nerf-n-layers', help='layers in nerf mlp', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--nerf-far', help='nerf far bound', metavar='FLOAT', type=click.FloatRange(min=1), default=16., show_default=True)
@click.option('--nerf-samples-per-ray', help='nerf samples per ray', type=click.IntRange(min=1), default=128, show_default=True)

# layout: loss arguments
@click.option('--concat-depth', help='depth to discriminator', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--concat-acc', help='acc to discriminator', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--recon-weight', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=1000., show_default=True)
@click.option('--aug-policy', help='aug policy', type=str, default='translation,color,cutout', required=True)
@click.option('--d-cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--d-cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--ray-lambda-finite-difference', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--ray-lambda-ramp-end', help='ramp function end (kimg)', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--use-wrapped-discriminator', help='wrap discriminator with second acc discriminator', type=bool, default=False, show_default=True)

# upsampler model: architecture
@click.option('--input-resolution', help='upsampler input resolution', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--concat-depth-and-acc', help='upsampler also generates depth, acc', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--use-3d-noise', help='sgan2 3D noise in upsampler', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--layout-model', help='layout model path', type=str)

# upsampler model: losses
@click.option('--lambda-rec', help='recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-up', help='upsample recon weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-gray-pixel', help='penalty on gray pixels where acc=1', metavar='FLOAT', type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--lambda-gray-pixel-falloff', help='penalty on gray pixels; exponential falloff rate', metavar='FLOAT', type=click.FloatRange(min=0), default=20, show_default=True)
@click.option('--D-ignore-depth-acc', help='ignores depth, acc inputs to discriminator', metavar='BOOL', type=bool, default=True, show_default=True)

# additional arguments for sky model
@click.option('--mask-prob', help='probability to use gt rgb in sky generator output', type=float)


def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=opts.z_dim, w_dim=opts.z_dim, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='external.stylegan.training.networks_stylegan2_terrain.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='external.stylegan.training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # check the training mode
    assert(opts.training_mode in ['layout', 'upsampler', 'sky'])

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.resolution = opts.img_resolution
    c.training_set_kwargs.depth_scale = opts.depth_scale
    c.training_set_kwargs.depth_clip = opts.depth_clip
    c.training_set_kwargs.use_disp = opts.use_disp
    c.training_set_kwargs.fov_mean = opts.fov_mean
    c.training_set_kwargs.fov_std = opts.fov_std
    c.training_set_kwargs.pose_path = opts.pose
    if opts.training_mode == 'layout':
        # additional sanity checks
        assert(opts.depth_scale == opts.nerf_far) # depth clip should match nerf far bound
        assert(opts.use_disp == True) # train it on disparity (inverse depth)

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2' and opts.training_mode == 'layout':
        c.G_kwargs.class_name = 'external.stylegan.training.networks_stylegan2_terrain.Generator'
        c.loss_kwargs.style_mixing_prob = 0 # 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 0 # 2 # Enable path length regularization.
        c.G_reg_interval = None # 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    elif opts.cfg == 'stylegan2' and opts.training_mode == 'upsampler':
        c.G_kwargs.class_name = 'external.stylegan.training.networks_stylegan2_terrain.Generator'
        c.loss_kwargs.style_mixing_prob = 0.0 # disabled; Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization. 
        c.G_reg_interval = 4 # Enable lazy regularization for G. 
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    elif opts.training_mode == 'sky':
        c.G_kwargs.class_name = 'external.stylegan.training.networks_stylegan3_sky.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # additional model arguments
    c.training_mode = opts.training_mode
    if opts.training_mode == 'layout':
        c.G_kwargs.img_resolution = opts.voxel_res
        c.G_kwargs.img_channels = opts.num_decoder_ch
        nerf_z_dim = opts.num_decoder_ch
        nerf_mlp_kwargs = dnnlib.EasyDict(
            class_name='external.gsn.models.generator.NerfStyleGenerator',
            n_layers=opts.nerf_n_layers,
            channels=128,
            out_channel=opts.nerf_out_ch,
            z_dim=nerf_z_dim,
        )
        generator_res = opts.voxel_res
        c.decoder_kwargs = dnnlib.EasyDict(
            class_name='external.gsn.models.generator.SceneGenerator',
            nerf_mlp_config=nerf_mlp_kwargs,
            img_res=opts.img_resolution,
            feature_nerf=opts.feature_nerf,
            global_feat_res=generator_res,
            coordinate_scale=generator_res * opts.voxel_size,
            alpha_activation='softplus',
            local_coordinates=True,
            hierarchical_sampling=False,
            density_bias=0,
            zfar_bias=False,
            use_disp=opts.use_disp,
            nerf_out_res=opts.nerf_out_res,
            samples_per_ray=opts.nerf_samples_per_ray,
            near=1,
            far=opts.nerf_far,
            alpha_noise_std=0,
        )
        c.torgb_kwargs = dnnlib.EasyDict(
            class_name='models.misc.networks.ToRGBTexture',
            in_channel=opts.nerf_out_ch,
        )
        c.wrapper_kwargs = dnnlib.EasyDict(
            voxel_res=generator_res, # opts.voxel_res,
            voxel_size=opts.voxel_size,
            img_res=opts.img_resolution,
            fov_mean=opts.fov_mean,
            fov_std=opts.fov_std,
        )
        c.loss_kwargs.loss_layout_kwargs = dnnlib.EasyDict(
            concat_depth = opts.concat_depth,
            concat_acc = opts.concat_acc,
            recon_weight = opts.recon_weight,
            aug_policy = opts.aug_policy,
            lambda_finite_difference=opts.ray_lambda_finite_difference,
            lambda_ramp_end = opts.ray_lambda_ramp_end * 1000, # convert to kimg
            use_wrapped_discriminator = opts.use_wrapped_discriminator
        )
    elif opts.training_mode == 'upsampler':
        c.G_kwargs.use_noise = True # helps improve texture
        c.G_kwargs.default_noise_mode = '3dnoise' if opts.use_3d_noise else 'random'
        c.G_kwargs.input_resolution = opts.input_resolution
        c.G_kwargs.num_additional_feature_channels = 2 if opts.concat_depth_and_acc else 0
        c.loss_kwargs.loss_upsampler_kwargs = dnnlib.EasyDict(
            d_ignore_depth_acc = opts.d_ignore_depth_acc,
            lambda_rec = opts.lambda_rec,
            lambda_up = opts.lambda_up,
            lambda_gray_pixel = opts.lambda_gray_pixel,
            lambda_gray_pixel_falloff = opts.lambda_gray_pixel_falloff,
        )
        c.wrapper_kwargs = dnnlib.EasyDict(
            layout_model_path = opts.layout_model,
        )
    elif opts.training_mode == 'sky':
        c.loss_kwargs.loss_sky_kwargs = dnnlib.EasyDict(
            mask_prob = opts.mask_prob
        )

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='external.stylegan.training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
