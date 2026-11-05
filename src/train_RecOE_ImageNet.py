# Source attribution (modified file).
# RecOE method code builds on the DivOE experimental framework:
# https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd/src
# RecOE changes include progressive subspaces, checkpoint handling, and scoring interfaces.
# DivOE-specific contributions have no verified upstream license.
# See ../THIRD_PARTY_NOTICES.md for attribution and the limits of the project LICENSE.

# train_RecOE_ImageNet.py
# ImageNet training script with progressive-rank dual Stiefel subspaces.
#
# Core idea:
#   Keep the original RecOE / RiemannianAdam objective unchanged, but optimize
#   U_id and U_ood with a progressive rank schedule:
#       h/2 -> h/2+h/4 -> h/2+h/4+h/8 -> ... -> h
#
# Expected local dependencies:
#   - models.resnet.resnet50
#   - imagenet21_aug.py
#   - utils.validation_dataset.validation_split
#   - DualSubspaceClass.py containing class DualSubspaceModule
#
# Example:
#   python train_RecOE_ImageNet.py \
#       --epochs 4 \
#       --batch_size 64 \
#       --oe_batch_size 64 \
#       --h_dim 512 \
#       --init_rank 256 \
#       --rank_expand_steps 1000 \
#       --expansion_mode half_remaining \
#       --stiefel_schedule stage_cosine \
#       --stiefel_lr 2 \
#       --stiefel_lr_min 1e-3 \
#       --stage_lr_decay_alpha 0.5 \
#       --stage_warmup_steps 50 \
#       --margin 5 \
#       --beta_ortho 0.1 \
#       --gamma 0.5 \
#       --learning_rate 1e-4 \
#       --ngpu 2

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from geoopt.optim import RiemannianAdam
from PIL import ImageFile

# Allow truncated images to be decoded in DataLoader workers.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.resnet import resnet50
from DualSubspaceClass import DualSubspaceModule
import imagenet21_aug
from imagenet_training_io import (
    SSDCachedImageLoader, loader_worker_options, move_training_batch,
    validate_imagenet_class_mapping,
)

try:
    BooleanOptionalAction = argparse.BooleanOptionalAction
except AttributeError:
    class BooleanOptionalAction(argparse.Action):
        """Backport argparse.BooleanOptionalAction for Python < 3.9."""

        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            expanded_option_strings = []
            for option_string in option_strings:
                expanded_option_strings.append(option_string)
                if option_string.startswith("--"):
                    expanded_option_strings.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=expanded_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from utils.validation_dataset import validation_split


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Tune ImageNet classifier with RecOE + progressive-rank dual Stiefel subspaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset roots.
    parser.add_argument("--imagenet_train_root", type=str, default="/mnt/data/imagenet2012/train")
    parser.add_argument("--imagenet_val_root", type=str, default="/mnt/data/imagenet2012/val")
    parser.add_argument(
        "--imagenet21k_root",
        type=str,
        default="/mnt/data/ImageNet-21K-P/imagenet21k_resized/imagenet21k_train",
    )

    # Training setup.
    parser.add_argument("--calibration", "-c", action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--oe_batch_size", type=int, default=64)
    parser.add_argument("--test_bs", type=int, default=200)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", "-d", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_train_steps_per_epoch",
        type=int,
        default=0,
        help="For hyperparameter search, cap train optimizer steps per epoch. <=0 uses the full epoch.",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=0,
        help="For hyperparameter search, cap ImageNet validation batches. <=0 evaluates the full validation set.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate every N epochs. 0 evaluates only the final epoch.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoints every N epochs. 0 saves only the final epoch.",
    )

    # Checkpoints.
    parser.add_argument("--save", type=str, default="./snapshots/RecOE_Progressive/imagenet")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint directory to resume from. Searches latest net_epoch_i.pt and stiefel_epoch_i.pt.",
    )
    parser.add_argument("--resume_epoch", type=int, default=-1)
    parser.add_argument("--test", "-t", action="store_true")

    # Acceleration.
    parser.add_argument("--ngpu", type=int, default=2, help="0 = CPU.")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="DataLoader workers per dataset (ID and OOD each use this many).")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=BooleanOptionalAction, default=True)
    parser.add_argument("--image_cache_root", type=str, default="",
                        help="Optional SSD cache of original image bytes; source datasets must be immutable.")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Print throughput and data-wait time every N steps; 0 disables step logs.")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--tf32", action="store_true",
                        help="Allow TF32 matrix multiplication on supported CUDA devices.")
    parser.add_argument("--amp_dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for forward/backward.")

    # Data augmentation.
    parser.add_argument(
        "--use_train_aug",
        action="store_true",
        help="Use random crop/flip and ImageNet21K RandAugment for training. If false, use deterministic resize.",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--randaugment_n", type=int, default=1)
    parser.add_argument("--randaugment_m", type=int, default=9)

    # Progressive-rank Stiefel options.
    parser.add_argument("--h_dim", type=int, default=512, help="final target subspace rank")
    parser.add_argument(
        "--init_rank",
        type=int,
        default=None,
        help="initial active rank. Default in module: ceil(h_dim/2).",
    )
    parser.add_argument(
        "--rank_expand_steps",
        type=int,
        default=1000,
        help="Expand rank every this many optimizer steps. <=0 disables step-based expansion.",
    )
    parser.add_argument(
        "--expansion_mode",
        type=str,
        default="half_remaining",
        choices=["half_remaining", "quarter_total", "one_by_one"],
    )
    parser.add_argument("--expand_on_epoch_end", action="store_true")
    parser.add_argument(
        "--force_full_rank_last_epoch",
        action=BooleanOptionalAction,
        default=True,
        help="At the beginning of the final epoch, expand directly to h_dim.",
    )

    # Original subspace loss options.
    parser.add_argument("--stiefel_lr", type=float, default=10.0)
    parser.add_argument("--stiefel_lr_min", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--beta_ortho", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.5)

    # Stiefel LR schedule.
    parser.add_argument(
        "--stiefel_schedule",
        type=str,
        default="stage_cosine",
        choices=["global_cosine", "stage_cosine"],
    )
    parser.add_argument(
        "--stage_lr_decay_alpha",
        type=float,
        default=0.5,
        help="stage_lr_max = stiefel_lr * (initial_active_rank / active_rank) ** alpha",
    )
    parser.add_argument("--stage_warmup_steps", type=int, default=50)

    return parser


def get_args(argv=None):
    return build_arg_parser().parse_args(argv)


def cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if total_steps <= 0:
        return float(lr_min)
    step = min(max(int(step), 0), int(total_steps))
    return float(lr_min + (lr_max - lr_min) * 0.5 * (1.0 + np.cos(step / total_steps * np.pi)))


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def build_stiefel_optimizer(stiefel: DualSubspaceModule, lr: float) -> RiemannianAdam:
    return RiemannianAdam(stiefel.parameters(), lr=float(lr), weight_decay=0)


def find_latest_imagenet_pair(load_dir: str, epoch: int = -1) -> Tuple[str, str, int]:
    def pair(i: int):
        return (
            os.path.join(load_dir, f"net_epoch_{i}.pt"),
            os.path.join(load_dir, f"stiefel_epoch_{i}.pt"),
        )

    if epoch >= 0:
        net_path, stiefel_path = pair(epoch)
        if not os.path.isfile(net_path):
            raise FileNotFoundError(net_path)
        if not os.path.isfile(stiefel_path):
            raise FileNotFoundError(stiefel_path)
        return net_path, stiefel_path, epoch

    for i in range(10000, -1, -1):
        net_path, stiefel_path = pair(i)
        if os.path.isfile(net_path) and os.path.isfile(stiefel_path):
            return net_path, stiefel_path, i

    raise FileNotFoundError(f"Could not find net_epoch_i.pt and stiefel_epoch_i.pt in {load_dir}")


def save_subspace_checkpoint(
    path: str,
    stiefel: DualSubspaceModule,
    args,
    z_dim: int,
    global_step: int,
    epoch: int,
    initial_active_rank: int,
    stage_start_step: int,
) -> None:
    # Persist active rank and EMA buffers together for matching evaluation.
    payload = {
        "state_dict": stiefel.state_dict(),
        "active_rank": int(stiefel.active_rank),
        "h_dim": int(args.h_dim),
        "z_dim": int(z_dim),
        "init_rank": None if args.init_rank is None else int(args.init_rank),
        "initial_active_rank": int(initial_active_rank),
        "expansion_mode": str(args.expansion_mode),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "stage_start_step": int(stage_start_step),
        "stiefel_schedule": str(args.stiefel_schedule),
        "stage_lr_decay_alpha": float(args.stage_lr_decay_alpha),
        "stage_warmup_steps": int(args.stage_warmup_steps),
    }
    torch.save(payload, path)


def load_subspace_checkpoint(
    path: str,
    z_dim: int,
    args,
    map_location="cpu",
) -> Tuple[DualSubspaceModule, Dict]:
    payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "state_dict" in payload:
        active_rank = int(payload.get("active_rank", args.init_rank or max(1, (args.h_dim + 1) // 2)))
        h_dim = int(payload.get("h_dim", args.h_dim))
        expansion_mode = str(payload.get("expansion_mode", args.expansion_mode))
        ckpt_z_dim = int(payload.get("z_dim", z_dim))
        if ckpt_z_dim != z_dim:
            raise ValueError(f"Checkpoint z_dim={ckpt_z_dim}, current z_dim={z_dim}")

        module = DualSubspaceModule(
            h_dim=h_dim,
            z_dim=z_dim,
            init_rank=active_rank,
            expansion_mode=expansion_mode,
        )
        module.load_state_dict(payload["state_dict"], strict=True)
        module.active_rank = active_rank
        return module, payload

    # Fallback for old plain state_dict checkpoints. This requires matching init_rank.
    module = DualSubspaceModule(
        h_dim=args.h_dim,
        z_dim=z_dim,
        init_rank=args.init_rank,
        expansion_mode=args.expansion_mode,
    )
    module.load_state_dict(payload, strict=True)
    return module, {}


def build_transforms(args):
    # Original ImageNet script normalizes to roughly [-1, 1].
    mean = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).tolist()
    std = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).tolist()

    if args.use_train_aug:
        train_transform_id = trn.Compose([
            trn.Resize((args.image_size, args.image_size)),
            trn.RandomCrop(args.image_size, padding=4),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])
        train_transform_ood = trn.Compose([
            imagenet21_aug.RandAugment(args.randaugment_n, args.randaugment_m, args=args),
            trn.Resize((args.image_size, args.image_size)),
            trn.RandomCrop(args.image_size, padding=4),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])
    else:
        train_transform_id = trn.Compose([
            trn.Resize((args.image_size, args.image_size)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])
        train_transform_ood = trn.Compose([
            trn.Resize((args.image_size, args.image_size)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    test_transform = trn.Compose([
        trn.Resize((args.image_size, args.image_size)),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])

    return train_transform_id, train_transform_ood, test_transform


def main(
    args=None,
    *,
    subspace_factory=DualSubspaceModule,
    checkpoint_loader=load_subspace_checkpoint,
    checkpoint_saver=save_subspace_checkpoint,
    run_name_prefix="imagenet_progressive",
):
    """Shared training loop; defaults retain the dual-subspace experiment."""
    if args is None:
        args = get_args()

    print(f"This experiment starts from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    torch.manual_seed(1)
    np.random.seed(args.seed)
    if args.ngpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    device = torch.device("cuda" if args.ngpu > 0 and torch.cuda.is_available() else "cpu")

    train_transform_id, train_transform_ood, test_transform = build_transforms(args)

    if args.prefetch < 0 or args.prefetch_factor < 1 or args.log_every < 0:
        raise ValueError("prefetch/log_every must be >= 0 and prefetch_factor must be >= 1")

    def image_folder(name, root, transform):
        begin = time.perf_counter()
        print(f"Indexing {name}: {root}", flush=True)
        kwargs = {}
        if args.image_cache_root:
            kwargs["loader"] = SSDCachedImageLoader(root, args.image_cache_root)
        dataset = dset.ImageFolder(root=root, transform=transform, **kwargs)
        print(f"{name}: {len(dataset):,} images, index took {time.perf_counter() - begin:.1f}s", flush=True)
        return dataset

    train_data_in = image_folder("ID train", args.imagenet_train_root, train_transform_id)
    test_data = image_folder("ID validation", args.imagenet_val_root, test_transform)
    class_to_idx = validate_imagenet_class_mapping(train_data_in, test_data)
    print("ImageNet class mapping verified: 1000 matching train/validation classes.", flush=True)
    train_data_ood = image_folder("OOD train", args.imagenet21k_root, train_transform_ood)
    num_classes = 1000

    calib_indicator = ""
    if args.calibration:
        train_data_in, _ = validation_split(train_data_in, val_share=0.1)
        calib_indicator = "_calib"

    worker_options = loader_worker_options(
        args.prefetch, args.prefetch_factor, args.persistent_workers,
    )
    print(f"DataLoader: {args.prefetch} workers per dataset, prefetch_factor={args.prefetch_factor}, "
          f"persistent_workers={bool(worker_options.get('persistent_workers', False))}", flush=True)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        **worker_options,
        pin_memory=True,
        drop_last=True,
    )
    train_loader_out = torch.utils.data.DataLoader(
        train_data_ood,
        batch_size=args.oe_batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        **worker_options,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        **worker_options,
        pin_memory=device.type == "cuda",
    )

    net = resnet50(pretrained=True)
    base_net = unwrap(net)
    z_dim = base_net.nChannels

    if args.resume:
        net_path, stiefel_path, resume_epoch = find_latest_imagenet_pair(args.resume, args.resume_epoch)
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        stiefel, resume_payload = checkpoint_loader(stiefel_path, z_dim, args, map_location="cpu")

        start_epoch = resume_epoch
        global_step = int(resume_payload.get("global_step", max(0, resume_epoch - 1) * min(len(train_loader_in), len(train_loader_out))))
        initial_active_rank = int(resume_payload.get("initial_active_rank", args.init_rank or stiefel.active_rank))
        stage_start_step = int(resume_payload.get("stage_start_step", global_step))

        print(f"Resumed net from: {net_path}")
        print(f"Resumed stiefel from: {stiefel_path}")
        print(
            f"Resume checkpoint epoch={resume_epoch}, start_epoch_index={start_epoch}, "
            f"global_step={global_step}, active_rank={stiefel.active_rank}/{args.h_dim}"
        )
    else:
        stiefel = subspace_factory(
            h_dim=args.h_dim,
            z_dim=z_dim,
            init_rank=args.init_rank,
            expansion_mode=args.expansion_mode,
        )
        start_epoch = 0
        global_step = 0
        initial_active_rank = int(stiefel.active_rank)
        stage_start_step = 0

    if args.ngpu > 1 and device.type == "cuda":
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    net.to(device)
    if args.channels_last:
        net.to(memory_format=torch.channels_last)
    stiefel.to(device)
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    full_train_steps = min(len(train_loader_in), len(train_loader_out))
    steps_per_epoch = full_train_steps
    if args.max_train_steps_per_epoch and args.max_train_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, int(args.max_train_steps_per_epoch))

    total_steps = args.epochs * steps_per_epoch

    # Backbone uses global cosine decay.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            total_steps,
            1.0,
            1e-6 / args.learning_rate,
        ),
    )

    if global_step > 0:
        for _ in range(min(global_step, total_steps)):
            scheduler.step()

    def stiefel_lr_at_step(step: int) -> float:
        if args.stiefel_schedule == "global_cosine":
            return cosine_annealing(
                step,
                total_steps,
                args.stiefel_lr,
                args.stiefel_lr_min,
            )

        active_rank = max(1, int(stiefel.active_rank))
        rank_scale = (float(initial_active_rank) / float(active_rank)) ** float(args.stage_lr_decay_alpha)
        stage_lr_max = float(args.stiefel_lr) * rank_scale

        if args.rank_expand_steps > 0:
            stage_len = max(1, int(args.rank_expand_steps))
        else:
            stage_len = max(1, steps_per_epoch)

        local_step = max(0, int(step) - int(stage_start_step))

        if args.stage_warmup_steps > 0 and local_step < args.stage_warmup_steps:
            warmup_ratio = float(local_step + 1) / float(args.stage_warmup_steps)
            return float(args.stiefel_lr_min + warmup_ratio * (stage_lr_max - args.stiefel_lr_min))

        progress = min(1.0, float(local_step) / float(stage_len))
        return float(
            args.stiefel_lr_min
            + (stage_lr_max - args.stiefel_lr_min) * 0.5 * (1.0 + np.cos(np.pi * progress))
        )

    current_stiefel_lr = stiefel_lr_at_step(global_step)
    opt_stiefel = build_stiefel_optimizer(stiefel, current_stiefel_lr)

    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use --amp_dtype float16")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    def maybe_expand_rank(reason: str, new_rank=None) -> None:
        nonlocal opt_stiefel, stage_start_step

        if stiefel.active_rank >= args.h_dim:
            return

        # Expansion replaces Parameters; the optimizer must bind the new objects.
        expanded = stiefel.expand_rank(new_rank)
        if expanded:
            stage_start_step = int(global_step)

            stiefel.to(device)
            current_lr = stiefel_lr_at_step(global_step)
            opt_stiefel = build_stiefel_optimizer(stiefel, current_lr)

            print(
                f"[Rank Expansion | {reason}] "
                f"step={global_step}, stage_start={stage_start_step}, "
                f"active_rank={stiefel.active_rank}/{args.h_dim}, "
                f"stiefel_lr={current_lr:.6g}, schedule={args.stiefel_schedule}"
            )

    def train_one_epoch(epoch: int):
        nonlocal global_step

        net.train()
        stiefel.train()

        # Keep metrics on device; .item() on every loss stalls the CPU/GPU pipeline.
        metric_sums = torch.zeros(5, device=device, dtype=torch.float64)
        stiefel_lr_avg = 0.0
        n_steps = 0
        n_images = 0
        data_wait_seconds = 0.0
        epoch_begin = time.perf_counter()
        previous_step_end = epoch_begin

        for step_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            if args.max_train_steps_per_epoch and step_idx >= args.max_train_steps_per_epoch:
                break

            data_wait_seconds += time.perf_counter() - previous_step_end
            data, target, b_in = move_training_batch(
                in_set, out_set, device, channels_last=args.channels_last,
            )
            n_images += len(data)

            optimizer.zero_grad(set_to_none=True)
            opt_stiefel.zero_grad(set_to_none=True)

            current_lr = stiefel_lr_at_step(global_step)
            set_optimizer_lr(opt_stiefel, current_lr)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                z, output = net(data, return_features=True)
                # Classification loss is computed on ID samples only.
                loss = F.cross_entropy(output[:b_in], target)

            # Keep normalization, reconstruction and manifold parameters in FP32.
            # AMP accelerates the backbone without reducing Stiefel precision.
            features = z.float() if amp_enabled else z
            stiefel_loss, stats = stiefel(
                features[:b_in], features[b_in:],
                beta_ortho=args.beta_ortho, margin=args.margin,
            )
            loss_sep = stats["loss_sep"]
            loss_ortho = stats["loss_ortho"]
            # CE supervises ID labels; gamma weights the complete subspace loss.
            total_loss = loss + args.gamma * stiefel_loss

            optimizer_step_taken = True
            if scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.scale(total_loss).backward()
                scaler.step(opt_stiefel)
                scaler.step(optimizer)
                scaler.update()
                # Overflow skips optimizer updates; do not advance LR/rank schedules.
                optimizer_step_taken = scaler.get_scale() >= scale_before
            else:
                total_loss.backward()
                opt_stiefel.step()
                optimizer.step()

            if optimizer_step_taken:
                scheduler.step()
                global_step += 1
            n_steps += 1

            if (
                optimizer_step_taken
                and args.rank_expand_steps > 0
                and global_step % args.rank_expand_steps == 0
                and stiefel.active_rank < args.h_dim
            ):
                maybe_expand_rank(reason="step")

            metric_sums.add_(torch.stack([
                loss.detach(), stiefel_loss.detach(), loss_sep.detach(),
                loss_ortho.detach(), total_loss.detach(),
            ]).to(dtype=torch.float64))
            stiefel_lr_avg += float(current_lr)

            if args.log_every and (n_steps == 1 or n_steps % args.log_every == 0):
                means = (metric_sums / n_steps).cpu().tolist()
                elapsed = time.perf_counter() - epoch_begin
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {n_steps}/{steps_per_epoch} | "
                    f"{n_images / max(elapsed, 1e-9):.1f} images/s | "
                    f"data_wait={data_wait_seconds / max(elapsed, 1e-9):.1%} | "
                    f"loss={means[4]:.5f} | rank={stiefel.active_rank}/{args.h_dim}",
                    flush=True,
                )
            previous_step_end = time.perf_counter()

        denom = max(n_steps, 1)
        means = (metric_sums / denom).cpu().tolist()
        elapsed = time.perf_counter() - epoch_begin
        state["train_net_loss"] = means[0]
        state["train_stiefel_loss"] = means[1]
        state["train_loss_sep"] = means[2]
        state["train_loss_ortho"] = means[3]
        state["train_total_loss"] = means[4]
        state["train_data_wait_s"] = data_wait_seconds
        state["train_time_s"] = elapsed
        state["train_images_per_s"] = n_images / max(elapsed, 1e-9)
        state["train_stiefel_lr"] = stiefel_lr_avg / denom
        state["active_rank"] = int(stiefel.active_rank)
        state["global_step"] = int(global_step)
        state["stage_start_step"] = int(stage_start_step)

        net.eval()
        stiefel.eval()

    def test():
        net.eval()
        stiefel.eval()

        metric_sums = torch.zeros(2, device=device, dtype=torch.float64)
        n_batches = 0
        n_examples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if args.max_test_batches and batch_idx >= args.max_test_batches:
                    break
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if args.channels_last:
                    data = data.contiguous(memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    output = net(data)
                    loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1)
                metric_sums[0].add_(loss.detach())
                metric_sums[1].add_(pred.eq(target).sum())
                n_batches += 1
                n_examples += target.numel()

        loss_sum, correct = metric_sums.cpu().tolist()
        state["test_loss"] = loss_sum / max(n_batches, 1)
        state["test_accuracy"] = correct / max(n_examples, 1)

    if args.test:
        test()
        print(state)
        return

    os.makedirs(args.save, exist_ok=True)

    run_name = (
        f"{run_name_prefix}_epoch:{args.epochs}"
        f"-dim:{args.h_dim}"
        f"-init:{initial_active_rank}"
        f"-expand:{args.rank_expand_steps}_{args.expansion_mode}"
        f"-beta_ortho:{args.beta_ortho}"
        f"-margin:{args.margin}"
        f"-gamma:{args.gamma}"
        f"-net_lr:{args.learning_rate}"
        f"-stiefel_lr:{args.stiefel_lr}"
        f"-stiefel_schedule:{args.stiefel_schedule}"
        f"-stage_alpha:{args.stage_lr_decay_alpha}"
        f"-warmup:{args.stage_warmup_steps}"
    )

    save_subdir = os.path.join(args.save, run_name)
    os.makedirs(save_subdir, exist_ok=True)
    print(f"Checkpoints and args.json: {save_subdir}")

    with open(os.path.join(save_subdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    with open(os.path.join(save_subdir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)

    csv_path = os.path.join(save_subdir, "training_results.csv")
    with open(csv_path, "w") as f:
        f.write(
            "epoch,time(s),active_rank,global_step,stage_start_step,train_stiefel_lr,"
            "train_net_loss,train_stiefel_loss,train_loss_sep,train_loss_ortho,"
            "train_total_loss,test_loss,test_accuracy,train_data_wait_s,train_time_s,train_images_per_s\n"
        )

    print("Beginning Training")
    print(f"Initial active rank: {stiefel.active_rank}/{args.h_dim}")
    print(
        f"Stiefel LR schedule: {args.stiefel_schedule}, "
        f"initial_lr={stiefel_lr_at_step(global_step):.6g}, "
        f"stage_start_step={stage_start_step}\n"
    )

    # Epoch names are 1-based to match the original ImageNet script:
    #   stiefel_epoch_1.pt, net_epoch_1.pt, ...
    for epoch_idx in range(start_epoch, args.epochs):
        if args.force_full_rank_last_epoch and epoch_idx == args.epochs - 1 and stiefel.active_rank < args.h_dim:
            maybe_expand_rank(reason="final_epoch", new_rank=args.h_dim)

        begin_epoch = time.time()

        train_one_epoch(epoch_idx)

        if args.expand_on_epoch_end and stiefel.active_rank < args.h_dim:
            maybe_expand_rank(reason="epoch_end")

        epoch_num = epoch_idx + 1

        should_eval = (
            epoch_idx == args.epochs - 1
            or (args.eval_every > 0 and epoch_num % args.eval_every == 0)
        )
        if should_eval:
            test()
        else:
            state["test_loss"] = float("nan")
            state["test_accuracy"] = float("nan")

        should_save = (
            epoch_idx == args.epochs - 1
            or (args.save_every > 0 and epoch_num % args.save_every == 0)
        )
        if should_save:
            stiefel_path = os.path.join(save_subdir, f"stiefel_epoch_{epoch_num}.pt")
            checkpoint_saver(
                stiefel_path,
                stiefel,
                args=args,
                z_dim=z_dim,
                global_step=global_step,
                epoch=epoch_num,
                initial_active_rank=initial_active_rank,
                stage_start_step=stage_start_step,
            )

            net_path = os.path.join(save_subdir, f"net_epoch_{epoch_num}.pt")
            torch.save(unwrap(net).state_dict(), net_path)

        with open(csv_path, "a") as f:
            f.write(
                "%03d,%05d,%d,%d,%d,%0.8f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.5f,%0.5f,%0.3f,%0.3f,%0.3f\n"
                % (
                    epoch_num,
                    time.time() - begin_epoch,
                    state["active_rank"],
                    state["global_step"],
                    state["stage_start_step"],
                    state["train_stiefel_lr"],
                    state["train_net_loss"],
                    state["train_stiefel_loss"],
                    state["train_loss_sep"],
                    state["train_loss_ortho"],
                    state["train_total_loss"],
                    state["test_loss"],
                    state["test_accuracy"],
                    state["train_data_wait_s"],
                    state["train_time_s"],
                    state["train_images_per_s"],
                )
            )

        print(
            "Epoch {epoch:3d} | Time {time_s:5d} | Rank {rank:4d}/{h_dim:4d} | "
            "Step {step:6d} | StageStart {stage_start:6d} | StiefelLR {stiefel_lr:.6g} | "
            "Net Loss {net_loss:.4f} | Stiefel Loss {stiefel_loss:.4f} | "
            "Sep {sep:.4f} | Ortho {ortho:.4f} | Total {total:.4f} | "
            "Test Acc {test_acc:.4f}".format(
                epoch=epoch_num,
                time_s=int(time.time() - begin_epoch),
                rank=state["active_rank"],
                h_dim=args.h_dim,
                step=state["global_step"],
                stage_start=state["stage_start_step"],
                stiefel_lr=state["train_stiefel_lr"],
                net_loss=state["train_net_loss"],
                stiefel_loss=state["train_stiefel_loss"],
                sep=state["train_loss_sep"],
                ortho=state["train_loss_ortho"],
                total=state["train_total_loss"],
                test_acc=state["test_accuracy"],
            )
        )


if __name__ == "__main__":
    main()
