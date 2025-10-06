import logging
from influxdb import InfluxDBClient

from shared import PerChannel, BAND_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfluxWriter:
    def __init__(self, influx_url: str, influx_database: str, influx_username: str, influx_password: str):
        self.client = InfluxDBClient(host=influx_url, username=influx_username, password=influx_password, database=influx_database, ssl=True, verify_ssl=True)
        logger.info(f"Connected to InfluxDB at {influx_url}")

    async def write_to_influx(self, eeg_data: list[PerChannel], start_of_epoch: float, samples_per_epoch: int, sampling_rate: int):
        json_body = []

        for channel in eeg_data:
            time = int((start_of_epoch + (samples_per_epoch / sampling_rate * 1000)))

            fields = {}
            for power in BAND_NAMES:
                fields[power] = getattr(channel.bandPowers, power)

            # Retrieve and add complexity metrics from the complexity dictionary
            for metric, value in channel.complexity.items():
                fields[metric] = value

            fields["over_threshold"] = len(channel.overThresholdIndices)

            data_point = {
                "measurement": "brainwave_epoch",
                "tags": {
                    "channel": channel.channelName,
                },
                "fields": fields,
                "time": time
            }
            json_body.append(data_point)

        logger.info(f"Saving {len(json_body)} points to Influx")
        self.client.write_points(json_body, time_precision='ms')

    async def write_raw_to_influx(self, eeg_data: list[PerChannel], start_of_epoch: float, samples_per_epoch: int, sampling_rate: int):
        json_body = []

        for channel in eeg_data:
            for i, sample in enumerate(channel.raw):
                time = int(start_of_epoch + (i / sampling_rate * 1000))

                data_point = {
                    "measurement": "brainwave_raw",
                    "tags": {
                        "channel": channel.channelName,
                    },
                    "fields": {
                        "raw_data": sample
                    },
                    "time": time
                }
                json_body.append(data_point)

        logger.info(f"Saving {len(json_body)} raw data points to Influx")
        self.client.write_points(json_body, time_precision='ms')
