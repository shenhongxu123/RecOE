# Third-party attribution and license notices

Updated on 2026-11-05.

Copyright (c) 2026 RecOE contributors.

The root [MIT License](LICENSE) applies to original RecOE contributions and
the four released checkpoint files listed below. This grant covers rights
held by the RecOE contributors. Third-party material retains its own
copyright and license terms.

This distribution contains or adapts material from Outlier Exposure,
torchvision, pytorch-randaugment, ASH, Google UDA, TensorFlow TPU, DARTS,
scikit-learn, and the DivOE experimental framework. The mapping below
identifies the upstream sources and links to the bundled license texts,
which preserve their original notices and terms.

## File-to-source mapping

Sources use a commit permalink where available or the reviewed version tag.

| Local code | Upstream source | Upstream license / notice |
| --- | --- | --- |
| `src/models/resnet.py`, base ResNet blocks and architecture | [torchvision ResNet, v0.13.1](https://github.com/pytorch/vision/blob/v0.13.1/torchvision/models/resnet.py) | [BSD-3-Clause](LICENSES/torchvision-BSD-3-Clause.txt); Copyright (c) Soumith Chintala 2016 |
| `src/models/resnet.py`, `ash_b`, `ash_p`, `ash_s` | [ASH](https://github.com/andrijazz/ash/blob/0528fd21a32fffde6a1ace08465f1d2879f6afd7/ash.py) | [MIT](LICENSES/ash-MIT.txt); Copyright (c) Andrija Djurisic |
| `src/models/wrn.py`, base WRN implementation | [Outlier Exposure WRN](https://github.com/hendrycks/outlier-exposure/blob/e6ede98a5474a0620d9befa50b38eaf584df4401/CIFAR/models/wrn.py) | [Apache-2.0](LICENSES/outlier-exposure-Apache-2.0.txt); also inherited through DivOE |
| `utils/display_results.py`, OOD metrics and result display | [Outlier Exposure metrics](https://github.com/hendrycks/outlier-exposure/blob/e6ede98a5474a0620d9befa50b38eaf584df4401/utils/display_results.py) | [Apache-2.0](LICENSES/outlier-exposure-Apache-2.0.txt); also inherited through DivOE |
| `utils/validation_dataset.py`, split helpers | [Outlier Exposure dataset helpers](https://github.com/hendrycks/outlier-exposure/blob/e6ede98a5474a0620d9befa50b38eaf584df4401/utils/validation_dataset.py) | [Apache-2.0](LICENSES/outlier-exposure-Apache-2.0.txt); definitions match this source |
| `utils/display_results.py`, cumulative-sum / ROC curve construction | [scikit-learn numerical helpers](https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/utils/extmath.py) and [ranking helpers](https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/metrics/_ranking.py) | [BSD-3-Clause](LICENSES/scikit-learn-BSD-3-Clause.txt); Copyright (c) 2007-2021 The scikit-learn developers |
| `src/imagenet21_aug.py`, RandAugment operations, `Lighting`, and `CutoutDefault` | [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment/blob/48b8f509c4bbda93bbe733d98b3fd052b6e4c8ae/RandAugment/augmentations.py) | [MIT](LICENSES/pytorch-randaugment-MIT.txt); Copyright (c) 2019 Ildoo Kim |
| `src/imagenet21_aug.py`, policy-list reference | [Google UDA policies](https://github.com/google-research/uda/blob/960684e363251772a5938451d4d2bc0f1da9e24b/image/randaugment/policies.py) | [Apache-2.0](LICENSES/uda-Apache-2.0.txt); Copyright 2019 The Google UDA Team Authors |
| `src/imagenet21_aug.py`, EfficientNet augmentation policy reference | [TensorFlow TPU AutoAugment](https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py) | [Apache-2.0](LICENSES/tensorflow-tpu-Apache-2.0.txt); Copyright 2017 The TensorFlow Authors |
| `src/imagenet21_aug.py`, `CutoutDefault` reference retained by upstream | [DARTS Cutout](https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/utils.py) | [Apache-2.0](LICENSES/darts-Apache-2.0.txt); Copyright (c) 2018, Hanxiao Liu |
| RecOE training/evaluation framework, WRN, augmentation, and inherited dataset/metric utilities | [DivOE inspected revision](https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd) | **No license file found for DivOE-specific contributions; see the unresolved scope below** |

## Modifications and provenance

RecOE adds dual-subspace objectives, progressive-rank training, checkpoint and
scoring interfaces, and related data-loading utilities. Existing model files
also retain feature-extraction interfaces and additional variants. This public
packaging adds annotations, standardizes entry-point/class names, and removes
the standalone fixed-rank entry points.

The attribution review compared upstream definitions with the retained code.
The WRN `BasicBlock` and `NetworkBlock`, the dataset-split helpers, several OOD
metric helpers, the base torchvision residual blocks, many RandAugment
operations, and all three ASH functions have matching definitions. These matches
establish attribution for those portions; they do not establish a license for
additional changes introduced by another project. Source-level comments mark
modified files and point to this document. Existing source references and
docstrings are preserved.

`src/imagenet21_aug.py` contains several upstream references to related policy
implementations. Their notices are included conservatively even when a link
documents an algorithm reference rather than a direct copy of an entire file.
The scikit-learn notice similarly covers the lineage of cumulative-sum and
ROC-related helper code, rather than claiming that the complete metric file
comes from one scikit-learn release.

## Unresolved DivOE-specific permission

The inspected [original DivOE tree](https://github.com/ZFancy/DivOE/tree/722c3369e9bf32a870ec6d14f586c0782a787ddd)
contains no `LICENSE`, `COPYING`, or `NOTICE` file. The previous README linked
to [the tmlr-group fork](https://github.com/tmlr-group/DivOE); its repository page
identifies [ZFancy/DivOE](https://github.com/ZFancy/DivOE) as the parent project.
This release now cites the original project and preserves the fork as provenance.

The Apache-2.0 license of Outlier Exposure supports attribution of identifiable
OE portions. It does not establish permission for additions made specifically
by DivOE. In particular, `utils/tiny_imagenet_loader.py` matches the DivOE
source, and the original RecOE project acknowledges DivOE as its experimental
framework. These inherited contributions remain explicitly outside the rights
granted by the root project LICENSE unless their owners have authorized them.

**Before publishing the complete release, confirm redistribution permission
for DivOE-specific material or establish an independently licensed source for
those portions.** Adding a source link, acknowledgement, or project LICENSE
does not resolve this missing permission. No authorization has been inferred
from the repository being publicly accessible.

## Released checkpoint license

The following trained RecOE checkpoint files are distributed under the
[MIT License](LICENSE), Copyright (c) 2026 RecOE contributors.

- `src/snapshots/cifar10/cifar10_wrn_s1_recoe_tune_progressive_stiefel_epoch_9.pt`
- `src/snapshots/cifar10/cifar10_wrn_s1_recoe_tune_progressive_stiefel_subspace_epoch_9.pt`
- `src/snapshots/imagenet/net_epoch_4.pt`
- `src/snapshots/imagenet/stiefel_epoch_4.pt`

Applicable upstream rights and notices continue to apply to material derived
from upstream pretrained weights.

## Dependencies and datasets

Installed dependencies listed in `environment.yml` retain their own licenses.
Dependencies, datasets, and upstream backbone initialization weights are
obtained separately and remain subject to their applicable terms.

Keep the relevant copyright notices, license texts, modification notices, and
this attribution record when redistributing the corresponding material.
No upstream project or author is represented as endorsing RecOE.
