import json
import logging
import time
from datetime import datetime
import numpy as np
import antropy as ant

from brainflow import BoardShim, DataFilter, DetrendOperations, FilterTypes, WindowOperations, BrainFlowInputParams
from nptyping import NDArray, Float64
from traitlets import List

from shared import BandPowers, PerChannel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainflowInput:

    def __init__(self, board_id: int, channel_names: List[str], serial_port: str, samples_per_epoch: int):
        BoardShim.enable_dev_board_logger()
        BoardShim.set_log_level(0)
        BoardShim.release_all_sessions()

        self.board_id = board_id
        self.channel_names = channel_names
        self.serial_port = serial_port
        self.samples_per_epoch = samples_per_epoch
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)[:len(channel_names)]
        logger.info(f"EEG Channels: {self.eeg_channels}")
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.board = None

    def connect_to_board(self):
        logger.info("Connecting to board")
        BoardShim.release_all_sessions()
        params = BrainFlowInputParams()
        params.serial_port=self.serial_port
        self.board = BoardShim(self.board_id, params)
        logger.info("Connected to board")
        self.board.prepare_session()
        logger.info("Starting stream")
        self.board.start_stream()
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".brainflow.csv"
        logger.info(f"Writing to file {filename}")
        self.board.add_streamer(f"file://{filename}:w")
        logger.info("Stream started")

        #self.sample_buffer = np.empty((len(self.eeg_channels), 0), dtype=float)
        #self.start_of_epoch = datetime.now().timestamp() * 1000

    async def fetch_and_process_samples(self) -> list[PerChannel]:
        if self.board is None:
            return []

        cd = self.board.get_current_board_data(self.samples_per_epoch)
        logger.debug(f"There are {cd.shape} samples ready")
        if cd.shape[1] < self.samples_per_epoch:
            return []

        # Data from every channel
        all_data: NDArray[Float64] = self.board.get_board_data(self.samples_per_epoch)

        # Just the EEG channel data
        eeg_channel_data = all_data[self.eeg_channels]

        eeg_data: list[PerChannel] = []

        logger.info("Collected enough samples for epoch")
        start_time = time.time()

        for index, channel in enumerate(self.eeg_channels):
            channel_name = self.channel_names[index]
            raw: NDArray[Float64] = eeg_channel_data[index]
            filtered: NDArray[Float64] = raw.copy()

            if any(value is None for value in filtered):
                logger.warning('Filtered data contains None values')
                eeg_data.append(PerChannel(index, channel_name, raw, filtered, [],
                                           BandPowers(0, 0, 0, 0, 0), []))
                continue

            DataFilter.detrend(filtered, DetrendOperations.LINEAR)
            DataFilter.perform_bandpass(filtered, self.sampling_rate, 4.0, 45.0, 4,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(filtered, self.sampling_rate, 45.0, 80.0, 4,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

            fft_filtered_json = ''
            fft_raw_json = ''
            try:
                next_power_of_two = 2 ** np.ceil(np.log2(len(filtered)))
                padded = np.zeros(int(next_power_of_two))
                padded[:len(filtered)] = filtered
                fft = DataFilter.perform_fft(padded, WindowOperations.HAMMING)
                fft_filtered_json = [{"_real": x.real, "_img": x.imag} for x in fft]
            except Exception as e:
                logger.error(f"Error performing FFT: {e}")

            try:
                next_power_of_two = 2 ** np.ceil(np.log2(len(raw)))
                padded = np.zeros(int(next_power_of_two))
                padded[:len(raw)] = raw
                fft = DataFilter.perform_fft(padded, WindowOperations.HAMMING)
                fft_raw_json = [{"_real": x.real, "_img": x.imag} for x in fft]
            except Exception as e:
                logger.error(f"Error performing FFT: {e}")

            over_threshold_indices = [i for i, sample in enumerate(filtered) if abs(sample) > 30]

            # Capture all complexity signals supported by the Antropy library.
            # Will filter to the most useful later.
            complexity = {}
            try:
                x = filtered

                # Calculate and store various entropy and complexity measures
                complexity["permutation_entropy"] = ant.perm_entropy(x, normalize=True)
                complexity["spectral_entropy"] = ant.spectral_entropy(x, sf=self.sampling_rate, method='welch', normalize=True)
                complexity["svd_entropy"] = ant.svd_entropy(x, normalize=True)
                complexity["approximate_entropy"] = ant.app_entropy(x)
                complexity["sample_entropy"] = ant.sample_entropy(x)

                # Calculate and store Hjorth parameters
                mobility, complexity_val = ant.hjorth_params(x)
                complexity["hjorth_mobility"] = mobility
                complexity["hjorth_complexity"] = complexity_val

                # Calculate and store zero-crossings
                complexity["num_zero_crossings"] = ant.num_zerocross(x)

                # Calculate and store fractal dimensions and DFA
                complexity["petrosian_fd"] = ant.petrosian_fd(x)
                complexity["katz_fd"] = ant.katz_fd(x)
                complexity["higuchi_fd"] = ant.higuchi_fd(x)
                complexity["detrended_fluctuation_analysis"] = ant.detrended_fluctuation(x)

                # Skipping Lempel-Ziv as needs a binary string

            except Exception as e:
                logger.error(f"Error performing complexity: {e}")


            filtered_2d = np.reshape(filtered, (1, -1))  # Reshape 'raw' into a 2D array with 1 row
            band_powers_for_channel = DataFilter.get_avg_band_powers(filtered_2d, [0], self.sampling_rate, False)

            eeg_data.append(PerChannel(
                index, channel_name, raw.tolist(), filtered.tolist(), fft_raw_json, fft_filtered_json,
                BandPowers(*band_powers_for_channel[0]),
                over_threshold_indices,
                complexity
            ))

        execution_time = time.time() - start_time
        logger.info(f"Processed epoch in: {execution_time / 1000} ms")

        return eeg_data


    def close(self):
        if self.board:
            b = self.board
            self.board = None
            b.stop_stream()
            b.release_session()
