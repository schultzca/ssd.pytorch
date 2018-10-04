from itertools import product as product
from math import sqrt as sqrt
from typing import List, Union

import torch


class PriorBox:
    r"""Generates prior boxes given network architecture and prior box
    configuration parameters.

    The SSD algorithm uses a set of predetermined priors (anchor boxes)
    that are mapped to the output of convolutional bounding box regressors.
    In the paper they defined a specific tiling scheme that is implemented
    here.

    Given a base network there are going to be m feature maps in which
    convolutional classifiers and bounding box regressors will be applied.
    For each cell of each feature map k prior boxes will be generated. These
    will have a scale that's specific to the feature map and a variety of
    aspect ratios.

    The number of prior boxes per cell can be computed using the following
    formula

        k = len(aspect_ratios) + (optional) extra

    In the paper they generated two bounding boxes with the same aspect ratio
    and a slightly different scale.
    """

    def __init__(self, feature_maps: List[int], scales: List[float],
                 aspect_ratios: Union[List[int], List[List[int]]],
                 extra_aspect_ratio: bool, clip: bool, variance: List[int], **kwargs):
        super().__init__()
        self.feature_maps = feature_maps
        self.scales = scales

        self.aspect_ratios = aspect_ratios
        if any(isinstance(e, list) for e in self.aspect_ratios):
            assert len(self.aspect_ratios) == len(self.feature_maps)
        else:
            self.aspect_ratios = [self.aspect_ratios for _ in range(len(self.feature_maps))]
        self.extra_aspect_ratio = extra_aspect_ratio

        self.clip = clip

        self.variance = variance
        assert any(v > 0 for v in self.variance)

        self.priors = self._priors()

    def _priors(self):
        """Given prior box parameters generate tensor of prior box coordinates."""
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = f
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.scales[k]
                mean += [cx, cy, s_k, s_k]

                if self.extra_aspect_ratio:
                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * self.scales[k + 1])
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
