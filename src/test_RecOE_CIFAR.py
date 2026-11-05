# Source attribution (modified file).
# RecOE method code builds on the DivOE experimental framework:
# https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd/src
# RecOE changes include progressive subspaces, checkpoint handling, and scoring interfaces.
# DivOE-specific contributions have no verified upstream license.
# See ../THIRD_PARTY_NOTICES.md for attribution and the limits of the project LICENSE.

# test_RecOE_CIFAR.py
# Evaluation script for DualSubspaceModule checkpoints.
#
# This script matches train_RecOE_CIFAR.py:
#   <method_name>_epoch_<i>.pt
#   <method_name>_subspace_epoch_<i>.pt
#
# The progressive subspace checkpoint is a dict:
#   {
#       "state_dict": ...,
#       "active_rank": ...,
#       "h_dim": ...,
#       "z_dim": ...,
#       "global_step": ...,
#       "epoch": ...
#   }
#
# Example:
#   python test_RecOE_CIFAR.py \
#       --dataset cifar10 \
#       --method_name cifar10_wrn_s1_recoe_tune_progressive_stiefel \
#       --load ./snapshots/RecOE_Progressive/cifar10s1/epoch_10_dim_80_init_40_expand_500_half_remaining_beta_ortho_1e-05_margin_0.5_gamma_0.5_lr_0.005_stiefel_lr_5.0 \
#       --h_dim 80 \
#       --ngpu 1

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image as PILImage

from models.wrn import WideResNet
from DualSubspaceClass import DualSubspaceModule 

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.display_results import (
    show_performance,
    get_measures,
    print_measures,
    print_measures_with_std,
)


from utils.reconstruction_scoring import (
    SCORE_MODE_DESCRIPTIONS,
    add_score_mode_argument,
    reconstruction_score,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR OOD detector with progressive-rank Stiefel subspaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Setup.
    add_score_mode_argument(parser)
    parser.add_argument("--test_bs", type=int, default=200)
    parser.add_argument("--num_to_avg", type=int, default=1)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Evaluate validation OODs without evaluating the final test-OOD datasets.",
    )
    parser.add_argument(
        "--method_name",
        "-m",
        type=str,
        default="cifar10_wrn_s1_recoe_tune_progressive_stiefel",
        help=(
            "Checkpoint prefix. For train_RecOE_CIFAR.py default naming, use "
            "cifar10_wrn_s1_recoe_tune_progressive_stiefel or "
            "cifar100_wrn_s1_recoe_tune_progressive_stiefel."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cifar10", "cifar100"],
        help="ID dataset. If omitted, infer from --method_name.",
    )

    # Model / loading.
    parser.add_argument("--layers", default=40, type=int)
    parser.add_argument("--widen-factor", default=2, type=int)
    parser.add_argument("--droprate", default=0.3, type=float)
    parser.add_argument(
        "--load",
        "-l",
        type=str,
        default="./snapshots/RecOE_Progressive/cifar10s1",
        help="Exact directory containing the progressive checkpoints.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Checkpoint epoch to load. Use -1 to search latest epoch automatically.",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN convolution algorithm search. Disabled by default for evaluation.",
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Use native PyTorch convolutions on CUDA to diagnose cuDNN errors; may be slower.",
    )

    # Subspace metadata fallback.
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=80)
    parser.add_argument("--h_dim_id", "--u_id_dim", dest="h_dim_id", type=int, default=None,
                        help="U_id target rank fallback for plain state_dict checkpoints")
    parser.add_argument("--h_dim_ood", "--u_ood_dim", dest="h_dim_ood", type=int, default=None,
                        help="U_ood target rank fallback for plain state_dict checkpoints")
    parser.add_argument("--init_rank", type=int, default=None)
    parser.add_argument("--init_rank_id", type=int, default=None)
    parser.add_argument("--init_rank_ood", type=int, default=None)
    parser.add_argument(
        "--expansion_mode",
        type=str,
        default="half_remaining",
        choices=["half_remaining", "quarter_total", "one_by_one"],
    )

    # Scoring.
    parser.add_argument(
        "--score_sign",
        type=str,
        default="original",
        choices=["original", "positive_ood"],
        help=(
            "original: reproduce old protocol using get_measures(-in_score, -out_score). "
            "positive_ood: use get_measures(out_score, in_score), with OOD positive."
        ),
    )
    parser.add_argument(
        "--strict_ood_paths",
        action="store_true",
        help="Raise an error if an OOD dataset path is missing. Default: skip missing datasets.",
    )

    # Local dataset roots.
    parser.add_argument("--cifar_root", type=str, default="/mnt/data/cifarpy")
    parser.add_argument("--ood_root", type=str, default="/mnt/data/ood_data")

    args = parser.parse_args()
    if args.h_dim_id is None:
        args.h_dim_id = args.h_dim
    if args.h_dim_ood is None:
        args.h_dim_ood = args.h_dim
    return args


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def forward_features(model, data):
    return unwrap(model).forward_features(data)


def classifier_from_features(model, z):
    m = unwrap(model)
    return F.linear(z, m.fc.weight, m.fc.bias)


def infer_dataset_name_from_method_name(method_name: str) -> Optional[str]:
    if "cifar10_" in method_name:
        return "cifar10"
    if "cifar100_" in method_name:
        return "cifar100"
    return None


def get_dataset_config(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...], int, str]:
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        return mean, std, 10, "cifar10"
    if dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        return mean, std, 100, "cifar100"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def resolve_dataset_config(args) -> Tuple[Tuple[float, ...], Tuple[float, ...], int, str]:
    inferred = infer_dataset_name_from_method_name(args.method_name)

    if args.dataset is None:
        if inferred is None:
            raise ValueError(
                "Cannot infer dataset from --method_name. "
                "Pass --dataset cifar10 or --dataset cifar100 explicitly."
            )
        return get_dataset_config(inferred)

    if inferred is not None and inferred != args.dataset:
        print(
            f"[Warning] --dataset={args.dataset} overrides method_name-inferred dataset={inferred}."
        )

    return get_dataset_config(args.dataset)


def find_checkpoint_pair(load_dir: str, method_name: str, epoch: int) -> Tuple[str, str, int]:
    load_dir = os.path.expanduser(load_dir)

    def paths(i: int) -> Tuple[str, str]:
        net_path = os.path.join(load_dir, f"{method_name}_epoch_{i}.pt")
        subspace_path = os.path.join(load_dir, f"{method_name}_subspace_epoch_{i}.pt")
        return net_path, subspace_path

    if epoch >= 0:
        net_path, subspace_path = paths(epoch)
        if not os.path.isfile(net_path):
            raise FileNotFoundError(f"Backbone checkpoint not found: {net_path}")
        if not os.path.isfile(subspace_path):
            raise FileNotFoundError(f"Progressive subspace checkpoint not found: {subspace_path}")
        return net_path, subspace_path, epoch

    for i in range(1000, -1, -1):
        net_path, subspace_path = paths(i)
        if os.path.isfile(net_path) and os.path.isfile(subspace_path):
            return net_path, subspace_path, i

    raise FileNotFoundError(
        "Could not find checkpoint pair in:\n"
        f"  {load_dir}\n"
        "Expected files:\n"
        f"  {method_name}_epoch_<i>.pt\n"
        f"  {method_name}_subspace_epoch_<i>.pt\n"
        "If the checkpoint was saved in a nested save_message directory, pass that exact directory to --load."
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

    stripped = {}
    for k, v in state.items():
        stripped[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    model.load_state_dict(stripped, strict=True)


def load_progressive_subspace(
    path: str,
    args,
    z_dim: int,
    map_location="cpu",
) -> Tuple[DualSubspaceModule, dict]:
    # Construct the saved active-rank shapes before restoring basis tensors.
    payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        legacy_h_dim = int(payload.get("h_dim", args.h_dim))
        h_dim_id = int(payload.get("h_dim_id", legacy_h_dim))
        h_dim_ood = int(payload.get("h_dim_ood", legacy_h_dim))
        active_rank_id = int(payload.get("active_rank_id", state["U_id"].shape[1]))
        active_rank_ood = int(payload.get("active_rank_ood", state["U_ood"].shape[1]))
        expansion_mode = str(payload.get("expansion_mode", args.expansion_mode))
        ckpt_z_dim = int(payload.get("z_dim", z_dim))

        if ckpt_z_dim != z_dim:
            raise ValueError(
                f"Checkpoint z_dim={ckpt_z_dim}, but current model z_dim={z_dim}. "
                "Check WideResNet architecture arguments."
            )

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

    # Fallback: plain state_dict.
    # This only works if --init_rank matches the saved U_id/U_ood shape.
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
    return module, {"state_dict": payload}


def build_ood_loader(
    name: str,
    root: str,
    transform,
    batch_size: int,
    num_workers: int,
    strict: bool,
    shuffle: bool = True,
):
    if not os.path.isdir(root):
        msg = f"[Skip] {name}: missing directory {root}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    dataset = dset.ImageFolder(root=root, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def configure_cudnn(args):
    cudnn.enabled = not args.disable_cudnn
    cudnn.benchmark = bool(args.cudnn_benchmark and cudnn.enabled)
    print("cuDNN: enabled={}, benchmark={}, version={}".format(
        cudnn.enabled, cudnn.benchmark, cudnn.version()
    ))


def main():
    args = get_args()
    result_name = "{}[{}]".format(args.method_name, args.score_mode)
    print("OOD score mode: {} = {}".format(
        args.score_mode, SCORE_MODE_DESCRIPTIONS[args.score_mode]
    ))
    print("Metric positive class: {}".format(
        "OOD" if getattr(args, "out_as_pos", False) or args.score_sign == "positive_ood" else "ID"
    ))
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if args.ngpu > 0 and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    mean, std, num_classes, dataset_name = resolve_dataset_config(args)
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset_name == "cifar10":
        test_data = dset.CIFAR10(args.cifar_root, train=False, transform=test_transform, download=False)
    else:
        test_data = dset.CIFAR100(args.cifar_root, train=False, transform=test_transform, download=False)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True,
    )

    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    z_dim = net.nChannels

    net_path, subspace_path, loaded_epoch = find_checkpoint_pair(
        args.load,
        args.method_name,
        args.epoch,
    )

    load_backbone_flexible(net, net_path, map_location="cpu")
    subspace, subspace_payload = load_progressive_subspace(
        subspace_path,
        args=args,
        z_dim=z_dim,
        map_location="cpu",
    )

    print(f"Loaded backbone from: {net_path}")
    print(f"Loaded progressive subspace from: {subspace_path}")
    print(f"Restored epoch: {loaded_epoch}")
    if subspace_payload:
        print(
            "Subspace metadata: "
            f"rank_id={subspace_payload.get('active_rank_id', subspace.active_rank_id)}/"
            f"{subspace_payload.get('h_dim_id', subspace.h_dim_id)}, "
            f"rank_ood={subspace_payload.get('active_rank_ood', subspace.active_rank_ood)}/"
            f"{subspace_payload.get('h_dim_ood', subspace.h_dim_ood)}, "
            f"global_step={subspace_payload.get('global_step', 'unknown')}"
        )

    net.eval()
    subspace.eval()

    if args.ngpu > 1 and device.type == "cuda":
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    net.to(device)
    subspace.to(device)

    configure_cudnn(args)
    print("Evaluation device:", device)

    with torch.no_grad():
        U_id = subspace.U_id
        U_ood = subspace.U_ood

        print(
            f"Active ranks: id={subspace.active_rank_id}/{subspace.h_dim_id}, "
            f"ood={subspace.active_rank_ood}/{subspace.h_dim_ood}"
        )
        print(
            f"EMA initialized: "
            f"{bool(subspace.ema_initialized.item()) if hasattr(subspace, 'ema_initialized') else 'unknown'}"
        )
        print(f"U_id shape: {tuple(U_id.shape)}")
        print(f"U_ood shape: {tuple(U_ood.shape)}")
        print(f"Cross-subspace penalty ||U_id^T U_ood||_F^2: {float(subspace.orthogonality_penalty().cpu()):.6f}")
        print(f"U_id orthogonality:{float(torch.norm(U_id.t() @ U_id - torch.eye(U_id.shape[1], device=U_id.device), p='fro').cpu()):.6f}")
        print(f"U_ood orthogonality:{float(torch.norm(U_ood.t() @ U_ood - torch.eye(U_ood.shape[1], device=U_ood.device), p='fro').cpu()):.6f}")

        if (subspace.active_rank_id < subspace.h_dim_id
                or subspace.active_rank_ood < subspace.h_dim_ood):
            print(
                "[Warning] The loaded checkpoint is not full-rank yet. "
                "Evaluation will use the current active rank only."
            )

    ood_num_examples = len(test_data) // 5

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.detach().cpu().numpy()

    def get_ood_scores(loader, in_dist: bool = False):
        scores = []
        right_score = []
        wrong_score = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= ood_num_examples // args.test_bs and not in_dist:
                    break

                data = data.to(device, non_blocking=True)
                target_np = target.numpy().squeeze()

                z = forward_features(net, data)
                output = classifier_from_features(net, z)

                rec_error = reconstruction_score(z, subspace, args.score_mode)

                scores.append(to_np(rec_error))

                if in_dist:
                    smax = to_np(F.softmax(output, dim=1))
                    preds = np.argmax(smax, axis=1)
                    right_indices = preds == target_np
                    wrong_indices = np.invert(right_indices)

                    if right_indices.any():
                        right_score.append(-np.max(smax[right_indices], axis=1))
                    if wrong_indices.any():
                        wrong_score.append(-np.max(smax[wrong_indices], axis=1))

        if in_dist:
            return concat(scores).copy(), concat(right_score).copy(), concat(wrong_score).copy()

        return concat(scores)[:ood_num_examples].copy()

    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    denom = max(num_wrong + num_right, 1)
    print("Error Rate {:.2f}".format(100 * num_wrong / denom))

    print("\nUsing CIFAR-10 as typical data" if num_classes == 10 else "\nUsing CIFAR-100 as typical data")

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

            # Raw scores are OOD-oriented; select the metric positive class here.
            if args.score_sign == "original":
                measures = get_measures(-in_score, -out_score)
            else:
                measures = get_measures(out_score, in_score)

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

        if num_to_avg >= 5:
            print_measures_with_std(aurocs, auprs, fprs, result_name)
        else:
            print_measures(auroc, aupr, fpr, result_name)

    if not args.validation_only:
        ood_root = os.path.expanduser(args.ood_root)
    
        print("\n\nTexture Detection")
        loader = build_ood_loader(
            "Textures",
            os.path.join(ood_root, "dtd", "images"),
            trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
            args.test_bs,
            num_workers=4,
            strict=args.strict_ood_paths,
        )
        get_and_print_results(loader)
    
        print("\n\nPlaces365 Detection")
        loader = build_ood_loader(
            "Places365",
            os.path.join(ood_root, "places365"),
            trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
            args.test_bs,
            num_workers=2,
            strict=args.strict_ood_paths,
        )
        get_and_print_results(loader)
    
        print("\n\nLSUN_C Detection")
        loader = build_ood_loader(
            "LSUN_C",
            os.path.join(ood_root, "LSUN"),
            trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
            args.test_bs,
            num_workers=1,
            strict=args.strict_ood_paths,
        )
        get_and_print_results(loader)
    
        print("\n\nLSUN_Resize Detection")
        loader = build_ood_loader(
            "LSUN_Resize",
            os.path.join(ood_root, "LSUN_resize"),
            trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
            args.test_bs,
            num_workers=1,
            strict=args.strict_ood_paths,
        )
        get_and_print_results(loader)
    
        print("\n\niSUN Detection")
        loader = build_ood_loader(
            "iSUN",
            os.path.join(ood_root, "iSUN"),
            trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]),
            args.test_bs,
            num_workers=1,
            strict=args.strict_ood_paths,
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
            print("No OOD test dataset was evaluated.")
    
    if not (args.validate or args.validation_only):
        return

    # Validation / synthetic OODs.
    auroc_list, aupr_list, fpr_list = [], [], []

    print("\n\nUniform[-1,1] Noise Detection")
    dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    ood_data = torch.from_numpy(
        np.random.uniform(
            size=(ood_num_examples * args.num_to_avg, 3, 32, 32),
            low=-1.0,
            high=1.0,
        ).astype(np.float32)
    )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)
    get_and_print_results(loader)

    if dataset_name == "cifar10":
        aux_data = dset.CIFAR100(args.cifar_root, train=False, transform=test_transform, download=False)
    else:
        aux_data = dset.CIFAR10(args.cifar_root, train=False, transform=test_transform, download=False)

    class AvgOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))
            return self.dataset[i][0] / 2.0 + self.dataset[random_idx][0] / 2.0, 0

        def __len__(self):
            return len(self.dataset)

    print("\n\nArithmetic Mean of Random Image Pair Detection")
    loader = torch.utils.data.DataLoader(
        AvgOfPair(aux_data),
        batch_size=args.test_bs,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    get_and_print_results(loader)

    if dataset_name == "cifar10":
        raw_aux_data = dset.CIFAR10(args.cifar_root, train=False, transform=trn.ToTensor(), download=False)
    else:
        raw_aux_data = dset.CIFAR100(args.cifar_root, train=False, transform=trn.ToTensor(), download=False)

    class GeomMeanOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))
            x = torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])
            return trn.Normalize(mean, std)(x), 0

        def __len__(self):
            return len(self.dataset)

    print("\n\nGeometric Mean of Random Image Pair Detection")
    loader = torch.utils.data.DataLoader(
        GeomMeanOfPair(raw_aux_data),
        batch_size=args.test_bs,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )
    get_and_print_results(loader)

    loader = torch.utils.data.DataLoader(
        raw_aux_data,
        batch_size=args.test_bs,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True,
    )

    jigsaw = lambda x: torch.cat((
        torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1), x[:, 16:, :16]), 2),
        torch.cat((x[:, 16:, 16:], torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
    ), 1)

    raw_aux_data.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])
    print("\n\nJigsawed Images Detection")
    get_and_print_results(loader)

    speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
    raw_aux_data.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])
    print("\n\nSpeckle Noised Images Detection")
    get_and_print_results(loader)

    pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
    raw_aux_data.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])
    print("\n\nPixelate Detection")
    get_and_print_results(loader)

    rgb_shift = lambda x: torch.cat((
        x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
        x[2:, :, :],
        x[0:1, :, :],
    ), 0)
    raw_aux_data.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])
    print("\n\nRGB Ghosted/Shifted Image Detection")
    get_and_print_results(loader)

    invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, :], 1 - x[2:, :, :]), 0)
    raw_aux_data.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])
    print("\n\nInverted Image Detection")
    get_and_print_results(loader)

    print("\n\nMean Validation Results")
    if len(auroc_list) > 0:
        print_measures(
            np.mean(auroc_list),
            np.mean(aupr_list),
            np.mean(fpr_list),
            method_name=result_name,
        )
    else:
        print("No validation OOD dataset was evaluated.")


if __name__ == "__main__":
    main()
