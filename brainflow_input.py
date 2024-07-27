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

    def __init__(self, board_id: int, channel_names: List[str], serial_port: str, samples_per_epoch: int, streamer: str):
        BoardShim.enable_dev_board_logger()
        BoardShim.set_log_level(0)
        BoardShim.release_all_sessions()

        self.last_data_collected = None
        self.board_id = board_id
        self.channel_names = channel_names
        self.serial_port = "" if serial_port is None else serial_port
        self.samples_per_epoch = samples_per_epoch
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)[:len(channel_names)]
        logger.info(f"EEG Channels: {self.eeg_channels}")
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.board = None
        self.streamer = streamer
        self.buffer = {channel: [] for channel in self.eeg_channels}

    def connect_to_board(self):
        logger.info("Connecting to board")
        BoardShim.release_all_sessions()
        params = BrainFlowInputParams()
        params.serial_port=self.serial_port
        # params.ip_address="225.1.1.1"
        # params.ip_port=6677
        # params.master_board=0
        self.board = BoardShim(self.board_id, params)
        logger.info("Connected to board")
        self.board.prepare_session()
        logger.info("Starting stream")
        self.board.start_stream()
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".brainflow.csv"
        logger.info(f"Writing to file {filename}")
        self.board.add_streamer(f"file://{filename}:w")
        if self.streamer is not None:
            self.board.add_streamer(self.streamer)
        logger.info("Stream started")

        # self.eeg_channels = self.board.get_eeg_channels(self.board._master_board_id)
        # self.sampling_rate = self.board.get_sampling_rate(self.board._master_board_id)

        #self.sample_buffer = np.empty((len(self.eeg_channels), 0), dtype=float)
        #self.start_of_epoch = datetime.now().timestamp() * 1000

    async def fetch_and_process_samples(self) -> list[PerChannel]:
        if self.board is None:
            return []

        # while True:
        #     time.sleep(0.1)
        #     data = self.board.get_board_data()
        #     logger.info(f"After 100ms have {data.shape} samples")

        # Data from every channel
        # Note Brainflow delivers it in quite a bursty way, so cannot just wait for 1 second and process the data:
        # 2024-07-27 08:52:21,930 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,030 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,131 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,232 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,332 - INFO - After 100ms have (24, 120) samples
        # 2024-07-27 08:52:22,433 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,533 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,634 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,735 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:22,835 - INFO - After 100ms have (24, 120) samples
        # 2024-07-27 08:52:22,936 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:23,037 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:23,138 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:23,238 - INFO - After 100ms have (24, 0) samples
        # 2024-07-27 08:52:23,339 - INFO - After 100ms have (24, 121) samples

        all_data: NDArray[Float64] = self.board.get_board_data()
        data_collected = time.perf_counter()

        for index, channel in enumerate(self.eeg_channels):
            self.buffer[channel].extend(all_data[channel])

        samples_collected_per_channel = len(self.buffer[self.eeg_channels[0]])
        if not all(len(self.buffer[channel]) >= self.samples_per_epoch for channel in self.eeg_channels):
            logger.info(f"Not enough samples yet - have {samples_collected_per_channel} for first channel")
            return []

        # Just the EEG channel data
        eeg_channel_data = all_data[self.eeg_channels]

        eeg_data: list[PerChannel] = []

        if self.last_data_collected is not None:
            elapsed_ms = (data_collected - self.last_data_collected) * 1000
            logger.info(f"Collected enough samples for epoch ({samples_collected_per_channel}) in {elapsed_ms} ms")
        self.last_data_collected = data_collected
        start_time = time.perf_counter()

        for index, channel in enumerate(self.eeg_channels):
            channel_name = self.channel_names[index]
            raw: NDArray[Float64] = np.array(self.buffer[channel][:self.samples_per_epoch])
            filtered: NDArray[Float64] = raw.copy()

            # Remove processed samples from buffer
            self.buffer[channel] = self.buffer[channel][self.samples_per_epoch:]

            logger.info(f"Have {len(raw)} samples for channel {channel_name}")

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
                # magnitudes = np.abs(fft)
                # freqs = np.fft.fftfreq(len(fft), 1/self.sampling_rate)
                # fft_filtered_json = [{"freq": freqs[i], "mag": magnitudes[i]} for i in range(len(fft))]
                # print(fft_filtered_json)

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

        execution_time = time.perf_counter() - start_time
        logger.info(f"Processed epoch in: {execution_time * 1000} ms")

        return eeg_data


    def close(self):
        if self.board:
            b = self.board
            self.board = None
            b.stop_stream()
            b.release_session()
