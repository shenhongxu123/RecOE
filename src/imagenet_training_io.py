"""Data-loading helpers for ImageNet training (compatible with PyTorch 1.12)."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import warnings

import torch
from PIL import Image


def validate_imagenet_class_mapping(train_data, validation_data, expected_classes=1000):
    """Reject independently numbered or incomplete ImageNet class folders.

    ImageFolder enumerates each split's sorted directories independently. A
    missing training directory therefore shifts classifier targets for later
    WNIDs, even though the classifier still has 1,000 outputs.
    """
    train_map = train_data.class_to_idx
    validation_map = validation_data.class_to_idx
    missing = sorted(set(validation_map) - set(train_map))
    unexpected = sorted(set(train_map) - set(validation_map))
    shifted = sorted(
        name for name in set(train_map) & set(validation_map)
        if train_map[name] != validation_map[name]
    )
    expected_indices = set(range(expected_classes))
    if (
        len(train_map) != expected_classes
        or len(validation_map) != expected_classes
        or train_map != validation_map
        or set(train_map.values()) != expected_indices
    ):
        details = [
            "ImageNet class mapping mismatch: train={}, validation={}, expected={}.".format(
                len(train_map), len(validation_map), expected_classes,
            ),
            "ImageFolder labels must match the pretrained classifier and validation labels.",
            "Missing training classes ({}): {}".format(len(missing), ", ".join(missing) or "none"),
            "Unexpected training classes ({}): {}".format(len(unexpected), ", ".join(unexpected) or "none"),
            "Classes with different train/validation indices: {}.".format(len(shifted)),
        ]
        if shifted:
            name = shifted[0]
            details.append("Example: {} has train index {} but validation index {}.".format(
                name, train_map[name], validation_map[name],
            ))
        details.append("Restore the complete class folders before training; do not compact labels for a partial split.")
        raise ValueError("\n".join(details))
    return dict(train_map)


class SSDCachedImageLoader:
    """Cache original image bytes; apply transforms after decoding as usual.

    Source datasets must be immutable. Remove the corresponding cache namespace
    if source images are replaced in place. Cache misses retain normal loading
    behavior, including Pillow's configured truncated-image policy.
    An optional decoder receives the original path and image_bytes=payload;
    evaluation uses it to preserve strict decoding and explicit recovery logs.
    """

    def __init__(self, source_root, cache_root, reserve_bytes=20 * 1024 ** 3, decoder=None):
        self.source_root = os.path.abspath(os.path.expanduser(source_root))
        canonical_root = os.path.realpath(self.source_root)
        namespace = hashlib.sha256(canonical_root.encode("utf-8")).hexdigest()[:16]
        self.cache_root = Path(cache_root).expanduser() / namespace
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reserve_bytes = reserve_bytes
        self.decoder = decoder
        self._writes_until_check = 0
        self._can_write = True
        self._warned = False

    def _warn(self, message):
        if not self._warned:
            warnings.warn(message, RuntimeWarning)
            self._warned = True

    def _store(self, cached_path, payload):
        temporary = None
        try:
            if self._writes_until_check <= 0:
                self._can_write = (
                    shutil.disk_usage(str(self.cache_root)).free
                    >= self.reserve_bytes + len(payload)
                )
                self._writes_until_check = 64
            self._writes_until_check -= 1
            if not self._can_write:
                self._warn("SSD image cache is low on space; reading uncached images from source.")
                return
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(cached_path.parent), delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
            os.replace(temporary, str(cached_path))
            temporary = None
        except OSError as exc:
            self._warn("Cannot write SSD image cache; using source images: {}".format(exc))
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def __call__(self, path):
        relative = os.path.relpath(path, self.source_root)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            raise ValueError("Image is outside the configured dataset root: {}".format(path))
        cached_path = self.cache_root / relative
        try:
            with open(str(cached_path), "rb") as handle:
                payload = handle.read()
        except FileNotFoundError:
            with open(path, "rb") as handle:
                payload = handle.read()
            self._store(cached_path, payload)
        if self.decoder is not None:
            return self.decoder(path, image_bytes=payload)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")


def loader_worker_options(num_workers, prefetch_factor=2, persistent_workers=True):
    """PyTorch rejects worker-only settings when loading in the main process."""
    if num_workers <= 0:
        return {}
    return {
        "prefetch_factor": prefetch_factor,
        "persistent_workers": persistent_workers,
    }


def move_training_batch(in_set, out_set, device, channels_last=False):
    """Transfer pinned batches first; CPU concatenation would lose pinning."""
    data_in = in_set[0].to(device, non_blocking=True)
    data_out = out_set[0].to(device, non_blocking=True)
    data = torch.cat((data_in, data_out), dim=0)
    if channels_last:
        data = data.contiguous(memory_format=torch.channels_last)
    target = in_set[1].to(device, non_blocking=True)
    return data, target, len(in_set[0])
