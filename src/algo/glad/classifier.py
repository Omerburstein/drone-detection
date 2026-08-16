"""GLAD's appearance gate on motion candidates, loaded once instead of per box.

This is the last thing standing between motion clutter and the output: every
blob that survives segmentation and the motion-coherence test is resized to
32x32 and accepted only if this network calls it class 1 (MAV).

The network is upstream's `Net` — a LeNet, 2 conv + 3 FC, ~61k parameters —
used exactly as released. The only change is *when* the checkpoint is read.
`Functions.Mynet_infer` constructs `Net()` and `torch.load()`s the 248 KB file
on **every candidate box of every frame**, inside a loop that runs up to 50
times per frame. Hoisting it to a single instance is numerically identical (the
weights fully overwrite the random init, and `Net` has no dropout or batch norm)
and removes what is almost certainly the bulk of the motion branch's cost.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .vendor import CLASSIFIER_WEIGHTS, add_import_roots, trusted_torch_load, weights_dir

PATCH_SIZE = 32  # the input size the classifier was trained at
MAV_CLASS = 1  # class 1 is MAV, class 0 is clutter


class MotionGate:
    """Callable replacement for `Functions.Mynet_infer` holding one loaded net.

    Signature and return value match upstream exactly — a BGR crop in, a class
    index out — so it can be substituted into `MOD2` without touching it.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        self._net = net
        self._to_tensor = transforms.ToTensor()

    def __call__(self, src: np.ndarray) -> int:
        """Class index for one candidate crop: 1 for MAV, 0 for clutter."""
        image = cv2.resize(src, (PATCH_SIZE, PATCH_SIZE))
        # Upstream feeds the crop BGR, as read from the frame. Kept: the network
        # was trained on whatever ordering this produced.
        tensor = torch.unsqueeze(self._to_tensor(image), dim=0)
        with torch.no_grad():
            output = torch.squeeze(self._net(tensor))
            return int(torch.argmax(torch.softmax(output, dim=0)))


def load_gate(glad_weights: Path | None = None) -> MotionGate:
    """Build the gate from the released `Net_best.pth`."""
    add_import_roots()
    from Functions import Net  # noqa: PLC0415 -- needs sys.path set up first

    path = (glad_weights or weights_dir()) / CLASSIFIER_WEIGHTS
    net = Net()
    with trusted_torch_load():
        net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    net.eval()
    return MotionGate(net)
