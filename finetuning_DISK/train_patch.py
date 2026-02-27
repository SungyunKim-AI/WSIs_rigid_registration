import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import numpy as np
import torch, random, argparse, yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from disk.common import Logger
from disk.data import Patch_Dataset, HyReCoDataset
from disk.model import DISK, ConsistentMatcher, CycleMatcher, run_registration
from disk.loss import Reinforce, IdentityGridReward


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--experiment', '-exp', type=str, required=True)
    parser.add_argument(
        '--data_dir', type=str, default=None # TODO: Dataset directory
    )
    parser.add_argument(
        '--benchmark-data-dir', type=str, default=None # TODO: Benchmark dataset directory
    )
    parser.add_argument(
        '--threshold', type=float, default=2.0,
        help='Threshold for reward criterion'
    )
    parser.add_argument(
        '--lm-fp', type=float, default=-0.5,
        help='Penalty for false positive matches (FP)'
    )
    parser.add_argument(
        '--lm-roi-fp', type=float, default=-0.5,
        help='Penalty for ROI false positive matches (ROI-FP)'
    )
    parser.add_argument(
        '--lm-kp', type=float, default=-0.005,
        help='Penalty for keypoints (KP)'
    )
    parser.add_argument(
        '--entropy-loss-alpha', type=float, default=0.01,
        help='Alpha for the entropy loss',
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='The size of the batch',
    )
    parser.add_argument(
        '--substep', type=int, default=2,
        help=('Number of batches to accumulate gradients over. Can be increased to'
            ' compensate for smaller batches on GPUs with less VRAM'),
    )
    parser.add_argument(
        '--warmup', type=int, default=500,
        help='The first (pseudo) epoch can be much shorter, this avoids wasting time.'
    )
    parser.add_argument(
        '--height', type=int, default=512,
        help='We train on images resized to (height, width)',
    )
    parser.add_argument(
        '--width', type=int, default=512,
        help='We train on images resized to (height, width)',
    )
    parser.add_argument(
        '--n-epochs', type=int, default=30,
        help='Number of (pseudo) epochs to train for',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate',
    )
    parser.add_argument(
        '--window-size', type=int, default=8,
        help='Window size for the DISK',
    )
    parser.add_argument(
        '--desc-dim', type=int, default=128,
        help='Dimensionality of descriptors to produce. 128 by default',
    )
    parser.add_argument(
        '--kernel-size', type=int, default=3,
        help='Size of the kernel for the U-Net',
    )
    parser.add_argument(
        '--load', type=str, default="None",
        help='Path to a checkpoint to resume training from',
    )
    parser.add_argument(
        '--epoch-offset', type=int, default=0,
        help=('Start counting epochs from this value. Influences the annealing '
            'procedures, and is therefore useful when restarting from a '
            'checkpoint'),
    )

    # WSI Augmentation parameters
    parser.add_argument('--color-jitter', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--mix-channel', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--max-translate', type=float, default=0.20)
    parser.add_argument('--max-rotate', type=float, default=30)
    parser.add_argument('--scale-range', type=float, nargs=2, default=[0.85, 1.15])
    parser.add_argument('--elastic-weight', type=float, default=0.5)
    parser.add_argument('--alpha-range', type=float, nargs=2, default=[40, 70])
    parser.add_argument('--sigma-range', type=float, nargs=2, default=[8, 12])
    parser.add_argument('--grid-grid', type=int, nargs=2, default=[4, 4])
    parser.add_argument('--magnitude-range', type=float, nargs=2, default=[20, 40])
    args = parser.parse_args()

    args.save_dir = f'./outputs/{args.experiment}'
    os.makedirs(args.save_dir, exist_ok=True)
    config_dict = vars(args)
    config_save_path = f"{args.save_dir}/config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return args


def train(args):
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEV}')

    # create the feature extractor and descriptor
    disk = DISK(in_features=3, window=args.window_size, desc_dim=args.desc_dim, kernel_size=args.kernel_size)
    if args.load != 'None':
        state_dict = torch.load(args.load, map_location='cpu')['extractor']
        disk.load_state_dict(state_dict)
    disk = disk.to(DEV)
    
    # Create custom dataset
    train_dataset = Patch_Dataset(
        args.data_dir,
        image_size=args.height,
        color_jitter=args.color_jitter,
        mix_channel=args.mix_channel,
        max_translate=args.max_translate,
        max_rotate=args.max_rotate,
        scale_range=tuple(args.scale_range),
        elastic_weight=args.elastic_weight,
        alpha_range=tuple(args.alpha_range),
        sigma_range=tuple(args.sigma_range),
        grid_grid=tuple(args.grid_grid),
        magnitude_range=tuple(args.magnitude_range)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Patch_Dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    benchmark_dataset = HyReCoDataset(args.benchmark_data_dir)
    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger = Logger(args.save_dir)

    # set up the inference-time matching algorithm and validation metrics
    valtime_matcher = CycleMatcher()

    optim = torch.optim.Adam(disk.parameters(), lr=args.lr)

    for e in range(args.n_epochs):
        e += args.epoch_offset

        if e == 0:
            ramp = 0.
        elif e == 1:
            ramp = 0.1
        else:
            ramp = min(1., 0.1 + 0.2 * e)

        loss_fn = Reinforce(
            IdentityGridReward(
                lm_tp=1.,
                lm_fp=args.lm_fp * ramp,
                lm_roi_fp=args.lm_roi_fp*ramp,
                th=args.threshold
            ),
            lm_kp=args.lm_kp * ramp,
            alpha=args.entropy_loss_alpha
        )

        inverse_T = 15 + 35 * min(1., 0.05 * e)
        matcher = ConsistentMatcher(inverse_T=inverse_T).to(DEV)
        matcher.requires_grad_(False)

        # the main training loop
        epoch_train_stats = []
        disk.train()
        train_iter = iter(train_loader)
        for i in tqdm(range(len(train_loader)), desc=f"Training Epoch {e}", ncols=150):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            bitmaps, images = batch.to(DEV, non_blocking=True)
            bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])

            features_ = disk.features(bitmaps_, kind='rng')
            features = features_.reshape(*bitmaps.shape[:2])
            stats = loss_fn.accumulate_grad(images, features, matcher)
            del bitmaps, images, features

            if i % args.substep == args.substep - 1:
                optim.step()
                optim.zero_grad()

            for sample in stats.flat:
                logger.add_scalars(sample, prefix='train')
                epoch_train_stats.append(sample)

            if e == 0 and i == args.warmup:
                break

        if epoch_train_stats:
            epoch_train_avg = {k: np.mean([s[k] for s in epoch_train_stats]) for k in epoch_train_stats[0].keys()}
            logger.add_scalars_at_step(epoch_train_avg, step=e, prefix='train/epoch')

        torch.save({
            'extractor': disk.state_dict(),
        }, f'{args.save_dir}/disk-wsi-save-{e}.pth')

        # validation loop
        disk.eval()
        epoch_test_stats = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(benchmark_loader, desc=f"Benchmark Epoch {e}", ncols=150)):
                he_image = batch[0].to(DEV, non_blocking=True)
                ihc_image = batch[1].to(DEV, non_blocking=True)
                target = batch[2]
                tre_90s, tre_m, num_matches, num_inliers = run_registration(
                    disk, valtime_matcher,
                    he_image, ihc_image,
                    target['landmarks_he'], target['landmarks_ihc']
                )

                epoch_test_stats.append({
                    'tre_90s': tre_90s,
                    'tre_m': tre_m,
                    'num_matches': num_matches,
                    'num_inliers': num_inliers
                })

        if epoch_test_stats:
            epoch_test_avg = {
                'tre_meadian90s': np.median([s['tre_90s'] for s in epoch_test_stats]),
                'tre_meanofmeans': np.mean([s['tre_m'] for s in epoch_test_stats]),
                'num_matches': np.mean([s['num_matches'] for s in epoch_test_stats]),
                'num_inliers': np.mean([s['num_inliers'] for s in epoch_test_stats])
            }
            logger.add_scalars_at_step(epoch_test_avg, step=e, prefix='test/epoch/discrete')

        print(f"Epoch {e} completed. Model saved to {args.save_dir}/save-{e}.pth")


if __name__ == '__main__':
    args = get_args()
    random.seed(42)
    train(args)
