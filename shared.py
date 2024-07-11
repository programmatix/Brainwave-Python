from typing import List

from nptyping import Float64, NDArray


class BandPowers:
    def __init__(self, delta: float, theta: float, alpha: float, beta: float, gamma: float):
        self.delta = delta
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class PerChannel:
    def __init__(self, channel_idx: int, channel_name: str, raw: NDArray[Float64], filtered: NDArray[Float64],
                 fft: List[complex], band_powers: BandPowers, over_threshold_indices: List[int]):
        self.channelIdx = channel_idx
        self.channelName = channel_name
        self.raw = raw
        self.filtered = filtered
        self.fft = fft
        self.bandPowers = band_powers
        self.overThresholdIndices = []