# Source attribution (modified file).
# RecOE method code builds on the DivOE experimental framework:
# https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd/src
# RecOE changes include progressive subspaces, checkpoint handling, and scoring interfaces.
# DivOE-specific contributions have no verified upstream license.
# See ../THIRD_PARTY_NOTICES.md for attribution and the limits of the project LICENSE.

# -*- coding: utf-8 -*-
# test_RecOE_ImageNet.py
#
# ImageNet OOD evaluation for progressive-rank dual Stiefel subspaces.
#
# Compatible with train_RecOE_ImageNet.py checkpoints:
#   net_epoch_<i>.pt
#   stiefel_epoch_<i>.pt
#
# stiefel_epoch_<i>.pt may be either:
#   1. metadata checkpoint:
#        {
#          "state_dict": ...,
#          "active_rank": ...,
#          "h_dim": ...,
#          "z_dim": ...,
#          ...
#        }
#   2. plain state_dict fallback.
#
# Example:
#   python test_RecOE_ImageNet.py \
#       --load ./snapshots/RecOE_Progressive/imagenet/<run_dir> \
#       --method_name imagenet_resnet50_progressive_stiefel \
#       --h_dim 512 \
#       --ngpu 1

from __future__ import annotations

import argparse
from functools import partial
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn

from models.resnet import resnet50
from DualSubspaceClass import DualSubspaceModule

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from utils.display_results import (
    show_performance,
    get_measures,
    print_measures,
    print_measures_with_std,
)
from utils.image_loading import load_rgb_image
from imagenet_training_io import SSDCachedImageLoader


from utils.reconstruction_scoring import (
    SCORE_MODE_DESCRIPTIONS,
    add_score_mode_argument,
    reconstruction_score,
)

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ImageNet OOD detector with progressive-rank dual Stiefel subspaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation setup.
    add_score_mode_argument(parser)
    parser.add_argument("--test_bs", type=int, default=200)
    parser.add_argument("--num_to_avg", type=int, default=1)
    parser.add_argument("--max_id_batches", type=int, default=0, help="Cap ID validation batches for fast hyperparameter search. <=0 uses all ID batches.")
    parser.add_argument("--max_ood_batches", type=int, default=0, help="Cap OOD batches per dataset for fast hyperparameter search. <=0 uses the standard protocol size.")
    parser.add_argument("--method_name", "-m", type=str, default="imagenet_resnet50_progressive_stiefel")
    parser.add_argument("--use_xent", "-x", action="store_true", help="Use cross-entropy error-detection score instead of MSP.")
    parser.add_argument("--out_as_pos", action="store_true", help="Treat OOD samples as positive in get_measures.")
    parser.add_argument(
        "--score_sign",
        type=str,
        default="original",
        choices=["original", "positive_ood"],
        help=(
            "original: reproduce previous RecOE protocol with get_measures(-in_score, -out_score). "
            "positive_ood: use get_measures(out_score, in_score). "
            "--out_as_pos is kept as a backward-compatible alias of positive_ood."
        ),
    )

    # Model / checkpoint.
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--init_rank", type=int, default=None, help="Fallback only for plain state_dict checkpoints.")
    parser.add_argument(
        "--expansion_mode",
        type=str,
        default="half_remaining",
        choices=["half_remaining", "quarter_total", "one_by_one"],
        help="Fallback only for plain state_dict checkpoints.",
    )
    parser.add_argument("--load", type=str, default="./snapshots/RecOE_Progressive/imagenet")
    parser.add_argument("--epoch", type=int, default=-1, help="Checkpoint epoch to load. -1 means search latest.")

    # Data roots.
    parser.add_argument("--imagenet_val_root", type=str, default="/mnt/data/imagenet2012/val")
    parser.add_argument("--ood_root", type=str, default="/mnt/data/ood_data")
    parser.add_argument("--image_cache_root", type=str, default="",
                        help="Optional SSD cache of original image bytes, shared with training.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--strict_ood_paths", action="store_true", help="Raise error if an OOD path is missing.")
    image_decoding = parser.add_mutually_exclusive_group()
    image_decoding.add_argument(
        "--allow_truncated_images",
        action="store_true",
        help="Retry truncated PIL images with an explicit warning; unrecoverable images still fail.",
    )
    image_decoding.add_argument(
        "--strict_image_decoding",
        dest="allow_truncated_images", action="store_false",
        help="Require strict PIL decoding, including for the single-subspace evaluation entries.",
    )
    parser.set_defaults(allow_truncated_images=False)

    # Acceleration.
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument(
        "--cudnn_benchmark", action="store_true",
        help="Enable cuDNN algorithm search. Disabled by default for evaluation.",
    )
    parser.add_argument(
        "--disable_cudnn", action="store_true",
        help="Use native CUDA convolutions instead of cuDNN for troubleshooting.",
    )
    parser.add_argument("--seed", type=int, default=1)

    # Optional benchmark switches.
    parser.add_argument("--include_ssb_hard", action="store_true")
    parser.add_argument("--include_ninco", action="store_true")

    return parser


def get_args(argv=None):
    return build_arg_parser().parse_args(argv)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def find_latest_pair(load_dir: str, epoch: int = -1) -> Tuple[str, str, int]:
    load_dir = os.path.expanduser(load_dir)

    def pair(i: int) -> Tuple[str, str]:
        return (
            os.path.join(load_dir, f"net_epoch_{i}.pt"),
            os.path.join(load_dir, f"stiefel_epoch_{i}.pt"),
        )

    if epoch >= 0:
        net_path, stiefel_path = pair(epoch)
        if not os.path.isfile(net_path):
            raise FileNotFoundError(f"Backbone checkpoint not found: {net_path}")
        if not os.path.isfile(stiefel_path):
            raise FileNotFoundError(f"Subspace checkpoint not found: {stiefel_path}")
        return net_path, stiefel_path, epoch

    for i in range(10000, -1, -1):
        net_path, stiefel_path = pair(i)
        if os.path.isfile(net_path) and os.path.isfile(stiefel_path):
            return net_path, stiefel_path, i

    raise FileNotFoundError(
        "Could not find checkpoint pair in:\n"
        f"  {load_dir}\n"
        "Expected files:\n"
        "  net_epoch_<i>.pt\n"
        "  stiefel_epoch_<i>.pt"
    )


def load_backbone_flexible(model: torch.nn.Module, path: str, map_location="cpu") -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    # Fallback for accidental DataParallel prefixes.
    stripped = {}
    for k, v in state.items():
        stripped[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    model.load_state_dict(stripped, strict=True)


def _infer_rank_from_state_dict(state_dict: dict) -> Optional[int]:
    for key in ("U_id", "module.U_id"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[1])
    return None


def load_progressive_subspace(
    path: str,
    args,
    z_dim: int,
    map_location="cpu",
) -> Tuple[DualSubspaceModule, dict]:
    # Restore active-rank shapes together with the checkpoint EMA centers.
    payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        active_rank = int(payload.get("active_rank", _infer_rank_from_state_dict(state) or args.init_rank or max(1, (args.h_dim + 1) // 2)))
        h_dim = int(payload.get("h_dim", args.h_dim))
        expansion_mode = str(payload.get("expansion_mode", args.expansion_mode))
        ckpt_z_dim = int(payload.get("z_dim", z_dim))

        if ckpt_z_dim != z_dim:
            raise ValueError(
                f"Subspace checkpoint z_dim={ckpt_z_dim}, but current ResNet feature dim={z_dim}. "
                "Check the backbone architecture."
            )

        module = DualSubspaceModule(
            h_dim=h_dim,
            z_dim=z_dim,
            init_rank=active_rank,
            expansion_mode=expansion_mode,
        )

        # New progressive checkpoints should load strictly.
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError:
            # Compatibility with accidental module. prefixes.
            stripped = {}
            for k, v in state.items():
                stripped[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
            module.load_state_dict(stripped, strict=False)

        module.active_rank = active_rank
        return module, payload

    # Plain state_dict fallback, including old full-rank stiefel_epoch_i.pt.
    state = payload
    active_rank = _infer_rank_from_state_dict(state) or args.init_rank or args.h_dim

    module = DualSubspaceModule(
        h_dim=args.h_dim,
        z_dim=z_dim,
        init_rank=active_rank,
        expansion_mode=args.expansion_mode,
    )

    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError:
        # Old DualSubspaceModule may not contain ema_initialized buffer.
        stripped = {}
        for k, v in state.items():
            stripped[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
        module.load_state_dict(stripped, strict=False)

    module.active_rank = active_rank
    return module, {"state_dict": state, "active_rank": active_rank, "h_dim": args.h_dim, "z_dim": z_dim}


def build_test_transform(args):
    mean = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).tolist()
    std = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1).tolist()
    return trn.Compose([
        trn.Resize((args.image_size, args.image_size)),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])


def build_image_loader(root, cache_root="", allow_truncated=False):
    decoder = partial(load_rgb_image, allow_truncated=allow_truncated)
    if cache_root:
        return SSDCachedImageLoader(root, cache_root, decoder=decoder)
    return decoder


def build_ood_loader(
    name: str,
    root: str,
    transform,
    batch_size: int,
    num_workers: int,
    strict: bool,
    shuffle: bool = True,
    allow_truncated: bool = False,
    cache_root: str = "",
):
    if not os.path.isdir(root):
        msg = f"[Skip] {name}: missing directory {root}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    dataset = dset.ImageFolder(
        root=root,
        transform=transform,
        loader=build_image_loader(root, cache_root, allow_truncated),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def concat_or_empty(arrays):
    if len(arrays) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def configure_cudnn(args):
    cudnn.enabled = not args.disable_cudnn
    cudnn.benchmark = bool(args.cudnn_benchmark and cudnn.enabled)
    print("cuDNN: enabled={}, benchmark={}, version={}".format(
        cudnn.enabled, cudnn.benchmark, cudnn.version()
    ))


def main(args=None, *, subspace_loader=load_progressive_subspace):
    """Shared evaluation protocol; the default loader restores dual subspaces."""
    if args is None:
        args = get_args()
    result_name = "{}[{}]".format(args.method_name, args.score_mode)
    print("OOD score mode: {} = {}".format(
        args.score_mode, SCORE_MODE_DESCRIPTIONS[args.score_mode]
    ))
    print("Metric positive class: {}".format(
        "OOD" if getattr(args, "out_as_pos", False) or args.score_sign == "positive_ood" else "ID"
    ))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print(f"This experiment starts from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    print({k: v for k, v in args._get_kwargs()})
    print("Image decoding policy:", "retry truncated files with warnings" if args.allow_truncated_images else "strict")
    print("Image cache:", args.image_cache_root or "disabled")

    torch.manual_seed(1)
    np.random.seed(args.seed)
    if args.ngpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    device = torch.device("cuda" if args.ngpu > 0 and torch.cuda.is_available() else "cpu")

    test_transform = build_test_transform(args)
    test_data = dset.ImageFolder(
        root=args.imagenet_val_root,
        transform=test_transform,
        loader=build_image_loader(args.imagenet_val_root, args.image_cache_root, args.allow_truncated_images),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=False,
    )

    net = resnet50(num_classes=1000)
    z_dim = net.nChannels

    net_path, stiefel_path, loaded_epoch = find_latest_pair(args.load, args.epoch)
    print(f"Loading backbone from: {net_path}")
    load_backbone_flexible(net, net_path, map_location="cpu")

    print(f"Loading progressive subspace from: {stiefel_path}")
    stiefel, stiefel_meta = subspace_loader(
        stiefel_path,
        args=args,
        z_dim=z_dim,
        map_location="cpu",
    )

    print(f"Restored epoch: {loaded_epoch}")
    print(
        "Subspace metadata: "
        f"active_rank={stiefel_meta.get('active_rank', stiefel.active_rank)}, "
        f"h_dim={stiefel_meta.get('h_dim', stiefel.h_dim)}, "
        f"global_step={stiefel_meta.get('global_step', 'unknown')}, "
        f"stage_start_step={stiefel_meta.get('stage_start_step', 'unknown')}"
    )

    net.eval()
    stiefel.eval()

    if args.ngpu > 1 and device.type == "cuda":
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    net.to(device)
    stiefel.to(device)

    configure_cudnn(args)
    print("Evaluation device:", device)

    with torch.no_grad():
        print(f"Active rank: {stiefel.active_rank}/{stiefel.h_dim}")
        for basis_name in ("U_id", "U_ood"):
            if hasattr(stiefel, basis_name):
                print(f"{basis_name} shape: {tuple(getattr(stiefel, basis_name).shape)}")
        if hasattr(stiefel, "ema_initialized"):
            print(f"EMA initialized: {bool(stiefel.ema_initialized.item())}")
        if hasattr(stiefel, "U_id") and hasattr(stiefel, "U_ood"):
            print(f"Cross-subspace penalty ||U_id^T U_ood||_F^2: {float(stiefel.orthogonality_penalty().detach().cpu()):.6f}")
        if int(stiefel.active_rank) < int(stiefel.h_dim):
            print("[Warning] Loaded checkpoint is not full-rank yet; evaluation uses the current active rank.")

    # Following the previous ImageNet protocol: use 1/5 of ID validation size per OOD dataset.
    ood_num_examples = len(test_data) // 5
    to_np = lambda x: x.detach().cpu().numpy()

    def get_ood_scores(loader, in_dist: bool = False):
        score_list = []
        right_score_list = []
        wrong_score_list = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if in_dist and args.max_id_batches > 0 and batch_idx >= args.max_id_batches:
                    break
                if not in_dist:
                    ood_batch_limit = max(1, ood_num_examples // args.test_bs)
                    if args.max_ood_batches > 0:
                        ood_batch_limit = min(ood_batch_limit, args.max_ood_batches)
                    if batch_idx >= ood_batch_limit:
                        break

                data = data.to(device, non_blocking=True)
                target_np = target.numpy().squeeze()

                z, output = net(data, return_features=True)
                rec_error = reconstruction_score(z, stiefel, args.score_mode)
                score_list.append(to_np(rec_error))

                if in_dist:
                    smax = to_np(F.softmax(output, dim=1))
                    preds = np.argmax(smax, axis=1)
                    right_indices = preds == target_np
                    wrong_indices = np.invert(right_indices)

                    if args.use_xent:
                        xent_score = to_np(output.mean(1) - torch.logsumexp(output, dim=1))
                        if right_indices.any():
                            right_score_list.append(xent_score[right_indices])
                        if wrong_indices.any():
                            wrong_score_list.append(xent_score[wrong_indices])
                    else:
                        if right_indices.any():
                            right_score_list.append(-np.max(smax[right_indices], axis=1))
                        if wrong_indices.any():
                            wrong_score_list.append(-np.max(smax[wrong_indices], axis=1))

        if in_dist:
            return (
                concat_or_empty(score_list).copy(),
                concat_or_empty(right_score_list).copy(),
                concat_or_empty(wrong_score_list).copy(),
            )

        return concat_or_empty(score_list)[:ood_num_examples].copy()

    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    denom = max(num_wrong + num_right, 1)
    print("Accuracy {:.2f}".format(100 * num_right / denom))

    print("\n\nError Detection")
    show_performance(wrong_score, right_score, method_name=args.method_name)

    auroc_list, aupr_list, fpr_list = [], [], []

    def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
        if ood_loader is None:
            return

        aurocs, auprs, fprs = [], [], []
        out_score = None

        for _ in range(num_to_avg):
            out_score = get_ood_scores(ood_loader)

            # Use the same metric convention across checkpoints and score modes.
            if args.out_as_pos or args.score_sign == "positive_ood":
                measures = get_measures(out_score, in_score)
            else:
                measures = get_measures(-in_score, -out_score)

            aurocs.append(measures[0])
            auprs.append(measures[1])
            fprs.append(measures[2])

        print("sample scores:", in_score[:3], out_score[:3])
        auroc = np.mean(aurocs)
        aupr = np.mean(auprs)
        fpr = np.mean(fprs)

        auroc_list.append(auroc)
        aupr_list.append(aupr)
        fpr_list.append(fpr)

        if num_to_avg >= 2:
            print_measures_with_std(aurocs, auprs, fprs, result_name)
        else:
            print_measures(auroc, aupr, fpr, result_name)

    ood_root = os.path.expanduser(args.ood_root)

    if args.include_ssb_hard:
        print("\n\nSSB-Hard Detection")
        loader = build_ood_loader(
            "SSB-Hard",
            os.path.join(ood_root, "ssb_hard"),
            test_transform,
            args.test_bs,
            num_workers=args.prefetch,
            strict=args.strict_ood_paths,
            allow_truncated=args.allow_truncated_images,
            cache_root=args.image_cache_root,
        )
        get_and_print_results(loader)

    if args.include_ninco:
        print("\n\nNINCO Detection")
        loader = build_ood_loader(
            "NINCO",
            os.path.join(ood_root, "NINCO", "NINCO_OOD_classes"),
            test_transform,
            args.test_bs,
            num_workers=args.prefetch,
            strict=args.strict_ood_paths,
            allow_truncated=args.allow_truncated_images,
            cache_root=args.image_cache_root,
        )
        get_and_print_results(loader)

    print("\n\nTexture Detection")
    loader = build_ood_loader(
        "Textures",
        os.path.join(ood_root, "dtd", "images"),
        test_transform,
        args.test_bs,
        num_workers=args.prefetch,
        strict=args.strict_ood_paths,
        allow_truncated=args.allow_truncated_images,
        cache_root=args.image_cache_root,
    )
    get_and_print_results(loader)

    print("\n\nPlaces365 Detection")
    loader = build_ood_loader(
        "Places365",
        os.path.join(ood_root, "Places"),
        test_transform,
        args.test_bs,
        num_workers=args.prefetch,
        strict=args.strict_ood_paths,
        allow_truncated=args.allow_truncated_images,
        cache_root=args.image_cache_root,
    )
    get_and_print_results(loader)

    print("\n\niNaturalist Detection")
    loader = build_ood_loader(
        "iNaturalist",
        os.path.join(ood_root, "iNaturalist"),
        test_transform,
        args.test_bs,
        num_workers=args.prefetch,
        strict=args.strict_ood_paths,
        allow_truncated=args.allow_truncated_images,
        cache_root=args.image_cache_root,
    )
    get_and_print_results(loader)

    print("\n\nSUN Detection")
    loader = build_ood_loader(
        "SUN",
        os.path.join(ood_root, "SUN"),
        test_transform,
        args.test_bs,
        num_workers=args.prefetch,
        strict=args.strict_ood_paths,
        allow_truncated=args.allow_truncated_images,
        cache_root=args.image_cache_root,
    )
    get_and_print_results(loader)

    print("\n\nMean Test Results!!!!!")
    if len(auroc_list) > 0:
        print_measures(
            np.mean(auroc_list),
            np.mean(aupr_list),
            np.mean(fpr_list),
            method_name=result_name,
        )
    else:
        print("No OOD dataset was evaluated.")


if __name__ == "__main__":
    main()
