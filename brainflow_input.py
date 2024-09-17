import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import antropy as ant
import mne

from brainflow import BoardShim, DataFilter, DetrendOperations, FilterTypes, WindowOperations, BrainFlowInputParams
from nptyping import NDArray, Float64
from traitlets import List

from shared import BandPowers, PerChannel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainflowInput:

    def __init__(self, board_id: int, default_channel_names: List[str], serial_port: str, samples_per_epoch: int, streamer: str):
        BoardShim.enable_dev_board_logger()
        BoardShim.set_log_level(0)
        BoardShim.release_all_sessions()

        self.last_data_collected = None
        self.board_id = board_id
        self.channel_names = default_channel_names
        self.serial_port = "" if serial_port is None else serial_port
        self.samples_per_epoch = samples_per_epoch
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.board = None
        self.streamer = streamer

    def connect_to_board(self, channel_names: Optional[List[str]]):
        if channel_names is not None:
            self.channel_names = channel_names
        logger.info("Connecting to board with channels " + str(self.channel_names))
        BoardShim.release_all_sessions()
        params = BrainFlowInputParams()
        params.serial_port=self.serial_port
        # params.ip_address="225.1.1.1"
        # params.ip_port=6677
        # params.master_board=0
        self.board = BoardShim(self.board_id, params)
        logger.info("Connected to board")
        self.board.prepare_session()

        # Turn off all channels
        for i in range(len(self.channel_names), 8):
            logger.info("Turning off channel " + str(i))
            self.board.config_board(str(i))
        symbols = ['!', '@', '#', '$', '%', '^', '&', '*']
        for i in range(len(self.channel_names)):
            symbol = symbols[i]
            logger.info("Turning on channel " + str(i) + " with " + symbol)
            self.board.config_board(symbol)

        # Start recording to SD.  It pre-allocates the data file so we use 12 hours as a compromise (smaller than 24).
        logger.info("Starting recording to SD")
        self.board.config_board('K')

        logger.info("Starting stream")
        self.board.start_stream()
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".brainflow.csv"
        logger.info(f"Writing to file {filename}")
        self.board.add_streamer(f"file://{filename}:w")
        if self.streamer is not None:
            self.board.add_streamer(self.streamer)
        logger.info("Stream started")

        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)[:len(self.channel_names)]
        logger.info(f"EEG Channels: {self.eeg_channels}")
        self.buffer = {channel: [] for channel in self.eeg_channels}

    async def fetch_and_process_samples(self) -> list[PerChannel]:
        if self.board is None:
            return []

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
            # N.b. elapsed_ms will rarely be exactly 1000ms due to the burst nature of the data.  It can also be
            # under 1s since we may have a backlog of data in the buffer.
            logger.info(f"Collected enough samples for epoch ({samples_collected_per_channel}) in {elapsed_ms} ms")
        self.last_data_collected = data_collected
        start_time = time.perf_counter()

        for index, channel in enumerate(self.eeg_channels):
            channel_name = self.channel_names[index]

            raw: NDArray[Float64] = np.array(self.buffer[channel][:self.samples_per_epoch])
            mne_raw = raw.copy().reshape(1, -1)
            # Remove processed samples from buffer
            self.buffer[channel] = self.buffer[channel][self.samples_per_epoch:]

            # Convert to MNE
            info = mne.create_info(ch_names=[channel_name], sfreq=self.sampling_rate, ch_types='eeg')
            # Brainflow Cyton data in uV, MNE expects V
            mne_scaled = mne_raw / 1_000_000
            mne_raw_array = mne.io.RawArray(mne_scaled, info)

            print("Raw (orig): ", raw[0:3])
            print("Raw (MNE) : ", mne_raw_array.get_data(units="µV")[0][0:3])


            # FFT
            mne_raw_array_microvolts = mne.io.RawArray(mne_raw_array.get_data(units="µV"), mne_raw_array.info)
            spectrum = mne_raw_array_microvolts.compute_psd(fmax=120)
            psds_raw, freqs_raw = spectrum.get_data(return_freqs=True)
            fft_raw_json = {"freq": freqs_raw, "power": psds_raw[0]}

            filtered_raw = mne_raw_array.get_data(units="µV")[0]
            DataFilter.detrend(filtered_raw, DetrendOperations.LINEAR)
            # We get a cleaner signal if we remove most of delta, which we don't care about much during waking hours anyway
            low_cutoff = 4
            DataFilter.perform_bandpass(filtered_raw, self.sampling_rate, low_cutoff, 40.0, 4, FilterTypes.BUTTERWORTH, 0)
            DataFilter.perform_bandstop(filtered_raw, self.sampling_rate, 40.0, 62.0, 4, FilterTypes.BUTTERWORTH, 0)
            DataFilter.perform_bandstop(filtered_raw, self.sampling_rate, 0.0, low_cutoff, 4, FilterTypes.BUTTERWORTH, 0)
            mne_raw_array = mne.io.RawArray(filtered_raw.reshape(1, -1) / 1_000_000, info)

            # MNE filters seem to work much less well than Brainflow's, unclear why - removing
            # mne_raw_array.filter(l_freq=5, h_freq=40, fir_design='firwin')
            # mne_raw_array.notch_filter(np.arange(50, 100), filter_length='auto', phase='zero')
            filtered = mne_raw_array.get_data(units="µV")[0]

            # MNE produces clearer FFTs than Brainflow
            mne_raw_array_microvolts = mne.io.RawArray(mne_raw_array.get_data(units="µV"), mne_raw_array.info)
            spectrum_filtered = mne_raw_array_microvolts.compute_psd(fmax=120)
            psds_filtered, freqs_filtered = spectrum_filtered.get_data(return_freqs=True)
            fft_filtered_json = {"freq": freqs_filtered, "power": psds_filtered[0]}

            # Capturing wavelets is WIP
            # Define frequencies of interest (1 Hz to 30 Hz at 1 Hz intervals)
            # frequencies = np.arange(1, 31, 1)
            # # Define number of cycles in Morlet wavelet
            # n_cycles = frequencies / 2.  # Different number of cycles per frequency
            # power = mne.time_frequency.tfr_array_morlet(mne_raw_array.get_data().reshape(1, 1, -1), sfreq=self.sampling_rate, freqs=frequencies, n_cycles=n_cycles, output='power')
            #
            # tfr = power[0, 0, :, :]
            # wavelets_filtered_json = {"freq": frequencies.tolist(), "power": tfr.tolist()}
            #
            # plt.figure(figsize=(10, 6))
            # plt.imshow(tfr, aspect='auto', origin='lower', extent=[0, tfr.shape[1], frequencies[0], frequencies[-1]])
            # plt.colorbar(label='Power')
            # plt.xlabel('Time (samples)')
            # plt.ylabel('Frequency (Hz)')
            # plt.title('Time-Frequency Representation (TFR)')
            # plt.show()
            # logger.info(f"Have {len(raw)} samples for channel {channel_name}")

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
                # AKA SampEn as used in Automated Detection of Driver Fatigue Based on Entropy and Complexity Measures, Zhang, 2014
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
            b.config_board('j') # stop recording to SD
            b.stop_stream()
            b.release_session()
