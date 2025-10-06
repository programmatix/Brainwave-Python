from typing import List

from nptyping import Float64, NDArray


BAND_DEFINITIONS = [
    (0.4, 1.0, "sdelta"),
    (1.0, 4.0, "fdelta"),
    (4.0, 8.0, "theta"),
    (8.0, 12.0, "alpha"),
    (12.0, 16.0, "sigma"),
    (16.0, 30.0, "beta"),
]

BAND_NAMES = [band[2] for band in BAND_DEFINITIONS]

class BandPowers:
    def __init__(self, sdelta: float, fdelta: float, theta: float, alpha: float, sigma: float, beta: float):
        self.sdelta = sdelta
        self.fdelta = fdelta
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta

    def to_dict(self):
        return {
            "sdelta": self.sdelta,
            "fdelta": self.fdelta,
            "theta": self.theta,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "beta": self.beta,
        }


class PerChannel:
    def __init__(self, channel_idx: int, channel_name: str, raw: NDArray[Float64], filtered: NDArray[Float64],
                 fft_raw, fft_filtered, band_powers: BandPowers, over_threshold_indices: List[int], complexity):
        # Non-Pythonic names as matching existing JSON
        self.channelIdx = channel_idx
        self.channelName = channel_name
        self.raw = raw
        self.filtered = filtered
        self.fftRaw = fft_raw
        self.fftFiltered = fft_filtered
        self.bandPowers = band_powers
        self.overThresholdIndices = over_threshold_indices
        self.complexity = complexity
