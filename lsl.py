import logging
import traceback

from pylsl import StreamInfo, StreamOutlet, vectorf
from typing_extensions import List

from shared import PerChannel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LslWriter:
    def __init__(self, id: str, channels: List[str], sampling_rate: int):
        name = "brainwave-lsl"
        type = "EEG"
        num_channels = len(channels)
        logger.info(f"Creating LSL channel with name {name}, type {type}, id {id}, num_channels {num_channels}, sampling_rate {sampling_rate}")
        self.info = StreamInfo(name, type, num_channels, sampling_rate, "double64", id)
        chns = self.info.desc().append_child("channels")
        for chan_ix, label in enumerate(channels):
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            # ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
            ch.append_child_value("scaling_factor", "1")

        self.outlet = StreamOutlet(self.info)


    async def write_to_lsl(self, eeg_data: list[PerChannel], start_of_epoch: float, samples_per_epoch: int, sampling_rate: int):

        try:
            #print("now sending data...")
            # start_time = local_clock()
            # sent_samples = 0
            # while True:
            #     elapsed_time = local_clock() - start_time
            #     required_samples = int(sampling_rate * elapsed_time) - sent_samples
            #     for sample_ix in range(required_samples):
            logger.info("Sending samples to LSL")
            samples_sent = 0
            for x in range(len(eeg_data[0].raw)):
                sample = vectorf()
                for i in range(len(eeg_data)):
                        #logger.info("Adding sample " + str(eeg_data[i].raw[x]) + " of type " + str(type(eeg_data[i].raw[x])))
                    sample.append(eeg_data[i].raw[x])
                #logger.info("Pushing sample " + str(sample))
                self.outlet.push_sample(sample)
                samples_sent += 1
            logger.info("Sent %d samples to LSL", samples_sent)
            # sent_samples += required_samples
                # await asyncio.sleep(1 / sampling_rate)
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()