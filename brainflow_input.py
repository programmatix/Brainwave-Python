import logging
from datetime import datetime
import numpy as np

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

        max_buffer_size = self.samples_per_epoch * 20
        self.sample_buffer = np.zeros((len(self.eeg_channels), max_buffer_size))
        self.sample_count = np.zeros(len(self.eeg_channels), dtype=int)
        self.start_of_epoch = datetime.now().timestamp() * 1000

    async def fetch_and_process_samples(self) -> list[PerChannel]:
        if self.board is None:
            return []

        all_data: NDArray[Float64] = self.board.get_board_data()

        for index, channel in enumerate(self.eeg_channels):
            new_samples = all_data[channel]
            num_new_samples = len(new_samples)
            self.sample_buffer[index, self.sample_count[index]:self.sample_count[index] + num_new_samples] = new_samples
            self.sample_count[index] += num_new_samples

        logger.info(f"Collected {all_data.shape} samples")

        eeg_data: list[PerChannel] = []

        if np.any(self.sample_count >= self.samples_per_epoch):
            for index, channel in enumerate(self.eeg_channels):
                channel_name = self.channel_names[index]
                raw: NDArray[Float64] = self.sample_buffer[index, :self.samples_per_epoch]
                logger.info(f"raw = {raw.shape}")
                num_samples_to_shift = self.sample_count[index] - self.samples_per_epoch
                if num_samples_to_shift > 0:
                    self.sample_buffer[index, :num_samples_to_shift] = self.sample_buffer[index, self.samples_per_epoch:self.sample_count[index]]
                self.sample_count[index] -= self.samples_per_epoch
                logger.info(f"raw = {raw.shape} sample_buffer = {self.sample_buffer.shape} sample_count = {self.sample_count}")
                filtered = raw.copy()

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

                fft = []
                try:
                    next_power_of_two = 2 ** (len(filtered) - 1).bit_length()
                    padded = filtered + [0] * (next_power_of_two - len(filtered))
                    fft = DataFilter.perform_fft(padded, WindowOperations.HAMMING)
                except Exception as e:
                    logger.error(f"Error performing FFT: {e}")

                band_powers = DataFilter.get_avg_band_powers([filtered], [0], self.sampling_rate, True)[0]
                over_threshold_indices = [i for i, sample in enumerate(filtered) if abs(sample) > 30]

                eeg_data.append(PerChannel(
                    index, channel_name, raw, filtered, fft,
                    BandPowers(*band_powers),
                    over_threshold_indices
                ))

        return eeg_data


    def close(self):
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
