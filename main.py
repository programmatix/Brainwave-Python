import argparse
import asyncio
import json
import logging
import traceback
from datetime import datetime

from brainflow_input import BrainflowInput
from influx import InfluxWriter
from websocket import WebsocketHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def sleep(ms: int):
    return asyncio.sleep(ms / 1000)


async def test():
    print ("Hello")

async def run_brainflow():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--board_id', type=int, required=True, help='The Brainflow board ID to connect to')
    parser.add_argument('-c', '--channels', nargs='+', required=True, help='Specify channel names')
    parser.add_argument('-j', '--just_wait', action='store_true', help='Just waits, does not take commands or do any processing')
    parser.add_argument('-w', '--wait_for_commands', action='store_true', help='Wait for directions over websocket')
    parser.add_argument('-sp', '--serial_port', type=str, help='Serial port e.g. /dev/ttyUSB0 (Linux) or COM11 (Windows)')
    parser.add_argument('-wp', '--websocket_port', type=int, help='Websocket port')
    parser.add_argument('-f', '--save_to_brainflow_file', type=str, help="Save the raw unprocessed data to file")
    parser.add_argument('--mqtt_url', type=str, help='MQTT URL')
    parser.add_argument('--mqtt_username', type=str, help='MQTT username')
    parser.add_argument('--mqtt_password', type=str, help='MQTT password')
    parser.add_argument('--influx_url', type=str, help='InfluxDB URL')
    parser.add_argument('--influx_database', type=str, help='InfluxDB database')
    parser.add_argument('--influx_username', type=str, help='InfluxDB username')
    parser.add_argument('--influx_password', type=str, help='InfluxDB password')
    args = parser.parse_args()

    logger.info(f"Starting Brainflow with args: {args}")

    done = False
    samples_per_epoch = 250

    influx = None
    if args.influx_url:
        if not all([args.influx_url, args.influx_token, args.influx_org, args.influx_bucket]):
            logger.error("All InfluxDB parameters (URL, token, org, bucket) must be provided")
            return
        influx = InfluxWriter(args.influx_url, args.influx_bucket, args.influx_org, args.influx_token)
    brainflow_input = BrainflowInput(args.board_id, args.channels, args.serial_port, samples_per_epoch)

    def set_done_true():
        nonlocal done
        done = True

    websocket_handler = WebsocketHandler(lambda: brainflow_input.connect_to_board(),
                                         lambda: brainflow_input.close(),
                                         set_done_true)

    if args.websocket_port:
        logger.info("Starting websocket server")
        _ = asyncio.create_task(websocket_handler.start_websocket_server(args.websocket_port))

    logger.info('WaitForCommands: ' + str(args.wait_for_commands))

    if args.wait_for_commands == False:
        logger.info("Connect")
        brainflow_input.connect_to_board()

    while not done:
        if args.just_wait == True:
            await sleep(10)
            continue

        try:
            #logger.info("Waiting for samples")
            await sleep(samples_per_epoch)

            eeg_data = await brainflow_input.fetch_and_process_samples()

            if len(eeg_data) > 0:
                start_of_epoch = datetime.now().timestamp() * 1000

                # if mqtt_client:
                #     mqtt_client.publish('brainwave/eeg', json.dumps([
                #         {
                #             'channel': channel.channel_name,
                #             'delta': channel.band_powers.delta,
                #             'theta': channel.band_powers.theta,
                #             'alpha': channel.band_powers.alpha,
                #             'beta': channel.band_powers.beta,
                #             'gamma': channel.band_powers.gamma
                #         } for channel in eeg_data
                #     ]))

                _ = asyncio.create_task(websocket_handler.broadcast_websocket_message(json.dumps({
                    'address': 'eeg',
                    'data': [channel.__dict__ for channel in eeg_data]
                })))

                if influx:
                    _ = asyncio.create_task(influx.write_to_influx(eeg_data, start_of_epoch, samples_per_epoch, brainflow_input.sampling_rate))

        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            pass

    logger.info('Done')


if __name__ == "__main__":
    asyncio.run(run_brainflow())