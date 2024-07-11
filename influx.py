import logging
from influxdb import InfluxDBClient

from shared import PerChannel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfluxWriter:
    def __init__(self, influx_url: str, influx_database: str, influx_username: str, influx_password: str):
        self.client = InfluxDBClient(host=influx_url, username=influx_username, password=influx_password, database=influx_database)
        logger.info(f"Connected to InfluxDB at {influx_url}")

    async def write_to_influx(self, eeg_data: list[PerChannel], start_of_epoch: float, samples_per_epoch: int, sampling_rate: int):
        json_body = []
        for channel in eeg_data:
            # time = int((start_of_epoch + (samples_per_epoch / sampling_rate * 1000)) * 1_000_000)
            time = int((start_of_epoch + (samples_per_epoch / sampling_rate * 1000)) * 1_000_000)
            data_point = {
                "measurement": "brainwave_epoch",
                "tags": {
                    "channel": channel.channel_name,
                },
                "fields": {
                    "delta": channel.band_powers.delta,
                    "theta": channel.band_powers.theta,
                    "alpha": channel.band_powers.alpha,
                    "beta": channel.band_powers.beta,
                    "gamma": channel.band_powers.gamma,
                },
                "time": time
            }
            json_body.append(data_point)

        logger.info(f"Saving {len(json_body)} points to Influx")
        self.client.write_points(json_body)