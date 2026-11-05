# Source attribution (modified file).
# RecOE method code builds on the DivOE experimental framework:
# https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd/src
# RecOE changes include progressive subspaces, checkpoint handling, and scoring interfaces.
# DivOE-specific contributions have no verified upstream license.
# See ../THIRD_PARTY_NOTICES.md for attribution and the limits of the project LICENSE.

# train_RecOE_CIFAR.py
# Progressive-rank Riemannian/Stiefel training for RecOE with stage-wise Stiefel LR restart.
#
# Compared with ordinary progressive-rank training, this version keeps:
#   1. Backbone SGD: global cosine decay.
#   2. Stiefel/RiemannianAdam: stage-wise cosine schedule with optional warmup after every rank expansion.
#
# Required local file:
#   DualSubspaceClass.py
#
# Example:
#   python train_RecOE_CIFAR.py \
#       --dataset cifar10 \
#       --epochs 10 \
#       --h_dim 80 \
#       --init_rank 40 \
#       --rank_expand_steps 500 \
#       --expansion_mode half_remaining \
#       --stiefel_schedule stage_cosine \
#       --stiefel_lr 5 \
#       --stiefel_lr_min 0.001 \
#       --stage_lr_decay_alpha 0.5 \
#       --stage_warmup_steps 20 \
#       --gamma 0.5 \
#       --margin 0.5 \
#       --beta_ortho 0.00001

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from geoopt.optim import RiemannianAdam

from models.wrn import WideResNet
from DualSubspaceClass import DualSubspaceModule

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tiny_imagenet_loader import customTinyImageNet
from utils.validation_dataset import validation_split


def get_args():
    parser = argparse.ArgumentParser(
        description="Tune a CIFAR classifier with RecOE + progressive-rank Stiefel subspaces + stage LR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / model.
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--cifar_root", default="/mnt/data/cifarpy",
        help="Root directory for both CIFAR splits.",
    )
    parser.add_argument(
        "--tiny_imagenet_root", default="/mnt/data/tiny-imagenet-200/train",
        help="Tiny ImageNet training directory containing class subdirectories.",
    )
    parser.add_argument("--model", "-m", type=str, default="wrn", choices=["allconv", "wrn", "densenet"])
    parser.add_argument("--calibration", "-c", action="store_true")

    # Backbone optimization.
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.005)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--oe_batch_size", type=int, default=128)
    parser.add_argument("--test_bs", type=int, default=200)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", "-d", type=float, default=0.0005)

    # WRN architecture.
    parser.add_argument("--layers", default=40, type=int)
    parser.add_argument("--widen-factor", default=2, type=int)
    parser.add_argument("--droprate", default=0.3, type=float)

    # Paths.
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Checkpoint root. Defaults to ./snapshots/RecOE_Progressive/{dataset}s{seed}.",
    )
    parser.add_argument("--load", "-l", type=str, default="./snapshots/pretrained")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Optional progressive checkpoint directory to resume from. "
            "The script searches latest *_epoch_i.pt and *_subspace_epoch_i.pt."
        ),
    )
    parser.add_argument("--resume_epoch", type=int, default=-1)
    parser.add_argument("--test", "-t", action="store_true")

    # Acceleration.
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    # Progressive-rank Stiefel options.
    parser.add_argument("--h_dim", type=int, default=64, help="final target subspace rank")
    parser.add_argument("--h_dim_id", "--u_id_dim", dest="h_dim_id", type=int, default=None,
                        help="final target rank of U_id; defaults to --h_dim")
    parser.add_argument("--h_dim_ood", "--u_ood_dim", dest="h_dim_ood", type=int, default=None,
                        help="final target rank of U_ood; defaults to --h_dim")
    parser.add_argument(
        "--init_rank",
        type=int,
        default=None,
        help="initial active rank. Default inside DualSubspaceModule: ceil(h_dim/2).",
    )
    parser.add_argument("--init_rank_id", type=int, default=None,
                        help="initial active rank of U_id; defaults to --init_rank or ceil(h_dim_id/2)")
    parser.add_argument("--init_rank_ood", type=int, default=None,
                        help="initial active rank of U_ood; defaults to --init_rank or ceil(h_dim_ood/2)")
    parser.add_argument(
        "--rank_expand_steps",
        type=int,
        default=500,
        help="Expand rank every this many optimizer steps. <=0 disables step-based expansion.",
    )
    parser.add_argument(
        "--expansion_mode",
        type=str,
        default="half_remaining",
        choices=["half_remaining", "quarter_total", "one_by_one"],
        help=(
            "half_remaining: 40->60->70->75... for h_dim=80; "
            "quarter_total: 40->60->80 for h_dim=80; "
            "one_by_one: add one column each expansion."
        ),
    )
    parser.add_argument(
        "--expand_on_epoch_end",
        action="store_true",
        help="Also try one rank expansion at every epoch end.",
    )
    parser.add_argument(
        "--force_full_rank_last_epoch",
        action="store_true",
        help="At the beginning of the final epoch, expand directly to h_dim.",
    )

    # Original subspace loss options.
    parser.add_argument("--beta_ortho", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)

    # Stiefel optimizer LR options.
    parser.add_argument("--stiefel_lr", type=float, default=5)
    parser.add_argument("--stiefel_lr_min", type=float, default=1e-3)
    parser.add_argument(
        "--stiefel_schedule",
        type=str,
        default="stage_cosine",
        choices=["global_cosine", "stage_cosine"],
        help=(
            "global_cosine: monotonically decay Stiefel LR over the full run. "
            "stage_cosine: restart Stiefel LR after every rank expansion."
        ),
    )
    parser.add_argument(
        "--stage_lr_decay_alpha",
        type=float,
        default=0.5,
        help=(
            "For stage_cosine, stage_lr_max = stiefel_lr * "
            "(initial_active_rank / active_rank) ** stage_lr_decay_alpha."
        ),
    )
    parser.add_argument(
        "--stage_warmup_steps",
        type=int,
        default=20,
        help="For stage_cosine, linearly warm up Stiefel LR after each rank expansion.",
    )

    args = parser.parse_args()
    if args.h_dim_id is None:
        args.h_dim_id = args.h_dim
    if args.h_dim_ood is None:
        args.h_dim_ood = args.h_dim
    return args


def cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if total_steps <= 0:
        return float(lr_min)
    step = min(max(int(step), 0), int(total_steps))
    return float(lr_min + (lr_max - lr_min) * 0.5 * (1.0 + np.cos(step / total_steps * np.pi)))


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def forward_features(model, data):
    return unwrap(model).forward_features(data)


def classifier_from_features(model, z):
    m = unwrap(model)
    return F.linear(z, m.fc.weight, m.fc.bias)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def build_stiefel_optimizer(stiefel: DualSubspaceModule, lr: float) -> RiemannianAdam:
    return RiemannianAdam(stiefel.parameters(), lr=float(lr), weight_decay=0)


def find_latest_pair(load_dir: str, prefix: str, epoch: int = -1) -> Tuple[str, str, int]:
    def pair(i: int):
        return (
            os.path.join(load_dir, f"{prefix}_epoch_{i}.pt"),
            os.path.join(load_dir, f"{prefix}_subspace_epoch_{i}.pt"),
        )

    if epoch >= 0:
        net_path, subspace_path = pair(epoch)
        if not os.path.isfile(net_path):
            raise FileNotFoundError(net_path)
        if not os.path.isfile(subspace_path):
            raise FileNotFoundError(subspace_path)
        return net_path, subspace_path, epoch

    for i in range(1000, -1, -1):
        net_path, subspace_path = pair(i)
        if os.path.isfile(net_path) and os.path.isfile(subspace_path):
            return net_path, subspace_path, i

    raise FileNotFoundError(f"Could not find checkpoint pair in {load_dir} with prefix {prefix}")


def save_subspace_checkpoint(
    path: str,
    stiefel: DualSubspaceModule,
    args,
    z_dim: int,
    global_step: int,
    epoch: int,
    initial_active_rank: int,
    initial_active_rank_id: int,
    initial_active_rank_ood: int,
    stage_start_step: int,
) -> None:
    # Basis shapes change with rank, so persist rank metadata with their tensors.
    payload = {
        "state_dict": stiefel.state_dict(),
        "active_rank": int(stiefel.active_rank),
        "active_rank_id": int(stiefel.active_rank_id),
        "active_rank_ood": int(stiefel.active_rank_ood),
        "h_dim": int(max(args.h_dim_id, args.h_dim_ood)),
        "h_dim_id": int(args.h_dim_id),
        "h_dim_ood": int(args.h_dim_ood),
        "z_dim": int(z_dim),
        "init_rank": None if args.init_rank is None else int(args.init_rank),
        "init_rank_id": None if args.init_rank_id is None else int(args.init_rank_id),
        "init_rank_ood": None if args.init_rank_ood is None else int(args.init_rank_ood),
        "initial_active_rank": int(initial_active_rank),
        "initial_active_rank_id": int(initial_active_rank_id),
        "initial_active_rank_ood": int(initial_active_rank_ood),
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
        state = payload["state_dict"]
        legacy_h_dim = int(payload.get("h_dim", args.h_dim))
        h_dim_id = int(payload.get("h_dim_id", legacy_h_dim))
        h_dim_ood = int(payload.get("h_dim_ood", legacy_h_dim))
        active_rank_id = int(payload.get("active_rank_id", state["U_id"].shape[1]))
        active_rank_ood = int(payload.get("active_rank_ood", state["U_ood"].shape[1]))
        expansion_mode = str(payload.get("expansion_mode", args.expansion_mode))

        module = DualSubspaceModule(
            h_dim=max(h_dim_id, h_dim_ood),
            z_dim=z_dim,
            h_dim_id=h_dim_id,
            h_dim_ood=h_dim_ood,
            init_rank_id=active_rank_id,
            init_rank_ood=active_rank_ood,
            expansion_mode=expansion_mode,
        )
        module.load_state_dict(state, strict=True)
        return module, payload

    # Fallback for plain state_dict checkpoints. This requires matching init_rank.
    module = DualSubspaceModule(
        h_dim=args.h_dim,
        z_dim=z_dim,
        init_rank=args.init_rank,
        h_dim_id=args.h_dim_id,
        h_dim_ood=args.h_dim_ood,
        init_rank_id=args.init_rank_id,
        init_rank_ood=args.init_rank_ood,
        expansion_mode=args.expansion_mode,
    )
    module.load_state_dict(payload, strict=True)
    return module, {}


def main():
    args = get_args()

    if args.save is None:
        args.save = os.path.join(".", "snapshots", "RecOE_Progressive", f"{args.dataset}s{args.seed}")

    print(f"This experiment starts from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    torch.manual_seed(1)
    np.random.seed(args.seed)


    # Dataset transforms retain the source experiment settings.
    # Dataset.
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        train_data_in = dset.CIFAR10(args.cifar_root, train=True, transform=test_transform, download=False)
        test_data = dset.CIFAR10(args.cifar_root, train=False, transform=test_transform, download=False)
        num_classes = 10
    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])        
        train_data_in = dset.CIFAR100(args.cifar_root, train=True, transform=test_transform, download=False)
        test_data = dset.CIFAR100(args.cifar_root, train=False, transform=test_transform, download=False)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    calib_indicator = ""
    if args.calibration:
        train_data_in, _ = validation_split(train_data_in, val_share=0.1)
        calib_indicator = "_calib"

    tiny_imagenet = dset.ImageFolder(
        root=args.tiny_imagenet_root,
        transform=trn.Compose([
            trn.Resize(32),
            trn.CenterCrop(32),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]),
    )
    ood_data = customTinyImageNet(tiny_imagenet)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    train_loader_out = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.oe_batch_size,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True,
    )

    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    z_dim = net.nChannels

    save_info = "recoe_tune_progressive_stiefel"

    if args.resume:
        net_path, subspace_path, resume_epoch = find_latest_pair(
            args.resume,
            f"{args.dataset}{calib_indicator}_{args.model}_s{args.seed}_{save_info}",
            args.resume_epoch,
        )
        net.load_state_dict(torch.load(net_path, map_location="cpu"))
        stiefel, resume_payload = load_subspace_checkpoint(subspace_path, z_dim, args, map_location="cpu")

        start_epoch = resume_epoch + 1
        global_step = int(resume_payload.get("global_step", start_epoch * len(train_loader_in)))
        legacy_initial_rank = int(resume_payload.get("initial_active_rank", stiefel.active_rank))
        initial_active_rank_id = int(resume_payload.get("initial_active_rank_id", legacy_initial_rank))
        initial_active_rank_ood = int(resume_payload.get("initial_active_rank_ood", legacy_initial_rank))
        initial_active_rank = max(initial_active_rank_id, initial_active_rank_ood)
        stage_start_step = int(resume_payload.get("stage_start_step", global_step))

        print(f"Resumed backbone from: {net_path}")
        print(f"Resumed subspace from: {subspace_path}")
        print(
            f"Resume epoch={resume_epoch}, start_epoch={start_epoch}, "
            f"global_step={global_step}, stage_start_step={stage_start_step}, "
            f"ranks=id:{stiefel.active_rank_id}/{stiefel.h_dim_id}, "
            f"ood:{stiefel.active_rank_ood}/{stiefel.h_dim_ood}"
        )
    else:
        # Restore pretrained backbone.
        model_found = False
        if args.load != "":
            for i in range(1000 - 1, -1, -1):
                model_name = os.path.join(
                    args.load,
                    args.dataset + calib_indicator + "_" + args.model + "_pretrained_epoch_" + str(i) + ".pt",
                )
                if os.path.isfile(model_name):
                    net.load_state_dict(torch.load(model_name, map_location="cpu"))
                    print("Pretrained model restored! Epoch:", i)
                    print(f"Loaded from: {model_name}")
                    model_found = True
                    break
            if not model_found:
                raise FileNotFoundError("could not find pretrained model to restore")

        stiefel = DualSubspaceModule(
            h_dim=max(args.h_dim_id, args.h_dim_ood),
            z_dim=z_dim,
            init_rank=args.init_rank,
            h_dim_id=args.h_dim_id,
            h_dim_ood=args.h_dim_ood,
            init_rank_id=args.init_rank_id,
            init_rank_ood=args.init_rank_ood,
            expansion_mode=args.expansion_mode,
        )
        start_epoch = 0
        global_step = 0
        initial_active_rank_id = int(stiefel.active_rank_id)
        initial_active_rank_ood = int(stiefel.active_rank_ood)
        initial_active_rank = max(initial_active_rank_id, initial_active_rank_ood)
        stage_start_step = 0

    # Keep progressive Stiefel module on the main device only. It dynamically replaces
    # ManifoldParameters during rank expansion, which is safer without DataParallel wrapping.
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        stiefel.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    total_steps = args.epochs * len(train_loader_in)

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
        """
        Stiefel LR policy.

        global_cosine:
            One monotonic decay during the entire training.

        stage_cosine:
            After every rank expansion, reset local step to 0 and run:
                warmup -> cosine decay
            while the stage maximum LR is scaled by current active rank:
                lr_max(r) = stiefel_lr * (initial_active_rank / active_rank) ** alpha
        """
        if args.stiefel_schedule == "global_cosine":
            return cosine_annealing(
                step,
                total_steps,
                args.stiefel_lr,
                args.stiefel_lr_min,
            )

        active_rank = max(1, int(stiefel.active_rank_id + stiefel.active_rank_ood))
        initial_total_rank = max(1, int(initial_active_rank_id + initial_active_rank_ood))
        rank_scale = (float(initial_total_rank) / float(active_rank)) ** float(args.stage_lr_decay_alpha)
        stage_lr_max = float(args.stiefel_lr) * rank_scale

        if args.rank_expand_steps > 0:
            stage_len = max(1, int(args.rank_expand_steps))
        else:
            # If step-based expansion is disabled, use one epoch as a reasonable stage length.
            stage_len = max(1, len(train_loader_in))

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

    def ranks_full() -> bool:
        return stiefel.active_rank_id >= stiefel.h_dim_id and stiefel.active_rank_ood >= stiefel.h_dim_ood

    def maybe_expand_rank(reason: str, new_rank_id=None, new_rank_ood=None) -> None:
        nonlocal opt_stiefel, stage_start_step
        if ranks_full():
            return

        expanded = stiefel.expand_ranks(new_rank_id, new_rank_ood)
        if expanded:
            # Rank expansion replaces U_id/U_ood ManifoldParameters, so optimizer must be rebuilt.
            # stage_start_step resets the local LR schedule.
            stage_start_step = int(global_step)

            if args.ngpu > 0:
                stiefel.cuda()

            current_lr = stiefel_lr_at_step(global_step)
            opt_stiefel = build_stiefel_optimizer(stiefel, current_lr)

            print(
                f"[Rank Expansion | {reason}] "
                f"step={global_step}, stage_start={stage_start_step}, "
                f"ranks=id:{stiefel.active_rank_id}/{stiefel.h_dim_id}, "
                f"ood:{stiefel.active_rank_ood}/{stiefel.h_dim_ood}, "
                f"stiefel_lr={current_lr:.6g}, schedule={args.stiefel_schedule}"
            )

    def train_one_epoch(epoch: int):
        nonlocal global_step

        net.train()
        stiefel.train()

        loss_avg = 0.0
        stiefel_loss_avg = 0.0
        loss_sep_avg = 0.0
        loss_ortho_avg = 0.0
        total_loss_avg = 0.0
        stiefel_lr_avg = 0.0
        n_steps = 0

        train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

        for in_set, out_set in zip(train_loader_in, train_loader_out):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]

            if args.ngpu > 0:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            b_in = len(in_set[0])

            optimizer.zero_grad(set_to_none=True)
            opt_stiefel.zero_grad(set_to_none=True)

            z = forward_features(net, data)
            z_id = z[:b_in]
            z_ood = z[b_in:]

            stiefel_loss, stats = stiefel(
                z_id,
                z_ood,
                beta_ortho=args.beta_ortho,
                margin=args.margin,
            )
            loss_sep = stats["loss_sep"]
            loss_ortho = stats["loss_ortho"]

            output = classifier_from_features(net, z_id)
            loss = F.cross_entropy(output, target)

            # ID labels train the classifier; both batches train the subspaces.
            total_loss = loss + args.gamma * stiefel_loss
            total_loss.backward()

            current_lr = stiefel_lr_at_step(global_step)
            set_optimizer_lr(opt_stiefel, current_lr)

            opt_stiefel.step()
            optimizer.step()
            scheduler.step()

            global_step += 1
            n_steps += 1

            if (
                args.rank_expand_steps > 0
                and global_step % args.rank_expand_steps == 0
                and not ranks_full()
            ):
                maybe_expand_rank(reason="step")

            loss_avg += loss.detach().item()
            stiefel_loss_avg += stiefel_loss.detach().item()
            total_loss_avg += total_loss.detach().item()
            loss_sep_avg += loss_sep.detach().item()
            loss_ortho_avg += loss_ortho.detach().item()
            stiefel_lr_avg += float(current_lr)

        denom = max(n_steps, 1)
        state["train_net_loss"] = loss_avg / denom
        state["train_stiefel_loss"] = stiefel_loss_avg / denom
        state["train_total_loss"] = total_loss_avg / denom
        state["train_loss_sep"] = loss_sep_avg / denom
        state["train_loss_ortho"] = loss_ortho_avg / denom
        state["train_stiefel_lr"] = stiefel_lr_avg / denom
        state["active_rank"] = int(stiefel.active_rank)
        state["active_rank_id"] = int(stiefel.active_rank_id)
        state["active_rank_ood"] = int(stiefel.active_rank_ood)
        state["global_step"] = int(global_step)
        state["stage_start_step"] = int(stage_start_step)

        net.eval()
        stiefel.eval()

    def test():
        net.eval()
        stiefel.eval()

        loss_avg = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                if args.ngpu > 0:
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = net(data)
                loss = F.cross_entropy(output, target)

                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
                loss_avg += loss.detach().item()

        state["test_loss"] = loss_avg / len(test_loader)
        state["test_accuracy"] = correct / len(test_loader.dataset)

    if args.test:
        test()
        print(state)
        return

    save_message = (
        f"epoch_{args.epochs}_dim_id_{args.h_dim_id}_dim_ood_{args.h_dim_ood}"
        f"_init_id_{initial_active_rank_id}_init_ood_{initial_active_rank_ood}"
        f"_expand_{args.rank_expand_steps}_{args.expansion_mode}"
        f"_beta_ortho_{args.beta_ortho}_margin_{args.margin}_gamma_{args.gamma}"
        f"_lr_{args.learning_rate}_stiefel_lr_{args.stiefel_lr}"
        f"_stiefel_schedule_{args.stiefel_schedule}"
        f"_stage_alpha_{args.stage_lr_decay_alpha}"
        f"_warmup_{args.stage_warmup_steps}"
    )
    subdir = os.path.join(args.save, save_message)
    os.makedirs(subdir, exist_ok=True)

    with open(os.path.join(subdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    csv_path = os.path.join(
        subdir,
        args.dataset + calib_indicator + "_" + args.model + "_s" + str(args.seed)
        + "_" + save_info + "_training_results.csv",
    )

    with open(csv_path, "w") as f:
        f.write(
            "epoch,time(s),active_rank_id,active_rank_ood,global_step,stage_start_step,train_stiefel_lr,"
            "train_net_loss,train_stiefel_loss,train_loss_sep,train_loss_ortho,"
            "train_total_loss,test_loss,test_error(%)\n"
        )

    print("Beginning Training")
    print(
        f"Initial active ranks: id={stiefel.active_rank_id}/{stiefel.h_dim_id}, "
        f"ood={stiefel.active_rank_ood}/{stiefel.h_dim_ood}"
    )
    print(
        f"Stiefel LR schedule: {args.stiefel_schedule}, "
        f"initial_lr={stiefel_lr_at_step(global_step):.6g}, "
        f"initial_active_rank={initial_active_rank}, "
        f"stage_start_step={stage_start_step}\n"
    )

    for epoch in range(start_epoch, args.epochs):
        state["epoch"] = epoch

        if args.force_full_rank_last_epoch and epoch == args.epochs - 1 and not ranks_full():
            maybe_expand_rank(
                reason="final_epoch",
                new_rank_id=args.h_dim_id,
                new_rank_ood=args.h_dim_ood,
            )

        begin_epoch = time.time()

        train_one_epoch(epoch)

        if args.expand_on_epoch_end and not ranks_full():
            maybe_expand_rank(reason="epoch_end")

        test()

        net_path = os.path.join(
            subdir,
            args.dataset + calib_indicator + "_" + args.model + "_s" + str(args.seed)
            + "_" + save_info + "_epoch_" + str(epoch) + ".pt",
        )
        torch.save(unwrap(net).state_dict(), net_path)

        prev_net_path = os.path.join(
            subdir,
            args.dataset + calib_indicator + "_" + args.model + "_s" + str(args.seed)
            + "_" + save_info + "_epoch_" + str(epoch - 1) + ".pt",
        )
        if os.path.exists(prev_net_path) and epoch % 10 != 0:
            os.remove(prev_net_path)

        subspace_path = os.path.join(
            subdir,
            args.dataset + calib_indicator + "_" + args.model + "_s" + str(args.seed)
            + "_" + save_info + "_subspace_epoch_" + str(epoch) + ".pt",
        )
        save_subspace_checkpoint(
            subspace_path,
            stiefel,
            args=args,
            z_dim=z_dim,
            global_step=global_step,
            epoch=epoch,
            initial_active_rank=initial_active_rank,
            initial_active_rank_id=initial_active_rank_id,
            initial_active_rank_ood=initial_active_rank_ood,
            stage_start_step=stage_start_step,
        )

        prev_subspace_path = os.path.join(
            subdir,
            args.dataset + calib_indicator + "_" + args.model + "_s" + str(args.seed)
            + "_" + save_info + "_subspace_epoch_" + str(epoch - 1) + ".pt",
        )
        if os.path.exists(prev_subspace_path) and epoch % 10 != 0:
            os.remove(prev_subspace_path)

        with open(csv_path, "a") as f:
            f.write(
                "%03d,%05d,%d,%d,%d,%d,%0.8f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.5f,%0.2f\n"
                % (
                    epoch + 1,
                    time.time() - begin_epoch,
                    state["active_rank_id"],
                    state["active_rank_ood"],
                    state["global_step"],
                    state["stage_start_step"],
                    state["train_stiefel_lr"],
                    state["train_net_loss"],
                    state["train_stiefel_loss"],
                    state["train_loss_sep"],
                    state["train_loss_ortho"],
                    state["train_total_loss"],
                    state["test_loss"],
                    100 - 100.0 * state["test_accuracy"],
                )
            )

        print(
            "Epoch {epoch:3d} | Time {time_s:5d} | Rank ID {rank_id:3d}/{h_dim_id:3d} | "
            "Rank OOD {rank_ood:3d}/{h_dim_ood:3d} | "
            "Step {step:5d} | StageStart {stage_start:5d} | StiefelLR {stiefel_lr:.6g} | "
            "Net Loss {net_loss:.4f} | Stiefel Loss {stiefel_loss:.4f} | "
            "Sep {sep:.4f} | Ortho {ortho:.4f} | Total {total:.4f} | "
            "Test Loss {test_loss:.4f} | Test Error {test_error:.2f}%".format(
                epoch=epoch + 1,
                time_s=int(time.time() - begin_epoch),
                rank_id=state["active_rank_id"],
                h_dim_id=args.h_dim_id,
                rank_ood=state["active_rank_ood"],
                h_dim_ood=args.h_dim_ood,
                step=state["global_step"],
                stage_start=state["stage_start_step"],
                stiefel_lr=state["train_stiefel_lr"],
                net_loss=state["train_net_loss"],
                stiefel_loss=state["train_stiefel_loss"],
                sep=state["train_loss_sep"],
                ortho=state["train_loss_ortho"],
                total=state["train_total_loss"],
                test_loss=state["test_loss"],
                test_error=100 - 100.0 * state["test_accuracy"],
            )
        )


if __name__ == "__main__":
    main()
