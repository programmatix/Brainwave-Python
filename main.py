import argparse
import asyncio
import json
import logging
import traceback
from datetime import datetime
from xmlrpc.client import boolean

from brainflow_input import BrainflowInput
from influx import InfluxWriter
from json_format import CustomEncoder
from lsl import LslWriter
from shared import BandPowers
from websocket import WebsocketHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_brainflow():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--board_id', type=int, required=True, help='The Brainflow board ID to connect to')
    parser.add_argument('-c', '--channels', nargs='+', required=True, help='Specify channel names')
    parser.add_argument('-j', '--just_wait', action='store_true',
                        help='Just waits, does not take commands or do any processing')
    parser.add_argument('-w', '--wait_for_commands', action='store_true', help='Wait for directions over websocket')
    parser.add_argument('-sp', '--serial_port', type=str,
                        help='Serial port e.g. /dev/ttyUSB0 (Linux) or COM11 (Windows)')
    parser.add_argument('-wp', '--websocket_port', type=int, help='Websocket port')
    parser.add_argument('-spe', '--samples_per_epoch', type=int, default=250, help='Samples per epoch')
    parser.add_argument('-f', '--save_to_brainflow_file', type=str, help="Save the raw unprocessed data to file")
    parser.add_argument('-o', '--output_dir', type=str, default=".", help="Where to save files")
    parser.add_argument('--mqtt_url', type=str, help='MQTT URL')
    parser.add_argument('--mqtt_username', type=str, help='MQTT username')
    parser.add_argument('--mqtt_password', type=str, help='MQTT password')
    parser.add_argument('--influx_url', type=str, help='InfluxDB URL')
    parser.add_argument('--influx_database', type=str, help='InfluxDB database')
    parser.add_argument('--influx_username', type=str, help='InfluxDB username')
    parser.add_argument('--influx_password', type=str, help='InfluxDB password')
    parser.add_argument('--ssl_cert', type=str, help='SSL cert file for websocket server')
    parser.add_argument('--ssl_key', type=str, help='SSL key file for websocket server')
    parser.add_argument('--streamer', type=str, help='Will add a Brainflow streamer output, e.g. streaming_board://224.0.0.0:10000, that can then be read by programs like OpenBCI GUI')
    parser.add_argument('--lsl', type=boolean, help='Will add an LSL streamer output with name "Brainwave-LSL" and type "EEG", and the provided identifier')

    args = parser.parse_args()

    logger.info(f"Starting Brainflow with args: {args}")

    done = False
    samples_per_epoch = args.samples_per_epoch

    def emit_event_callback(event_name: str, timestamp: float):
        logger.info(f"Emitting event: {event_name} for {timestamp}")
        message = json.dumps({
            'address': 'brainflow_event',
            'event': event_name,
            'timestamp': timestamp
        })
        asyncio.create_task(websocket_handler.broadcast_websocket_message(message))

    influx = None
    if args.influx_url:
        if not all([args.influx_url, args.influx_database, args.influx_username, args.influx_password]):
            logger.error("All InfluxDB parameters (URL, token, org, bucket) must be provided")
            return
        influx = InfluxWriter(args.influx_url, args.influx_database, args.influx_username, args.influx_password)
    brainflow_input = BrainflowInput(args.board_id, args.channels, args.serial_port, samples_per_epoch, args.streamer, args.output_dir, emit_event_callback)

    lsl = None
    if args.lsl:
        lsl = LslWriter("cyton", args.channels, brainflow_input.sampling_rate)

    def set_done_true():
        nonlocal done
        done = True

    websocket_handler = WebsocketHandler(args.ssl_cert, args.ssl_key,
                                         brainflow_input.connect_to_board,
                                         lambda: brainflow_input.close(),
                                         set_done_true,
                                         emit_event_callback)

    websocket_server_task = None
    if args.websocket_port:
        logger.info("Starting websocket server")
        websocket_server_task = asyncio.create_task(websocket_handler.start_websocket_server(args.websocket_port))

    logger.info('WaitForCommands: ' + str(args.wait_for_commands))

    if args.wait_for_commands == False:
        logger.info("Connect")
        brainflow_input.connect_to_board(None)

    while not done:
        if args.just_wait == True:
            await asyncio.sleep(10 / 1000)
            continue

        try:
            await asyncio.sleep(samples_per_epoch / 1000)

            eeg_data = await brainflow_input.fetch_and_process_samples()

            if len(eeg_data) > 0:
                start_of_epoch = datetime.now().timestamp() * 1000

                _ = asyncio.create_task(websocket_handler.broadcast_websocket_message(json.dumps({
                    'address': 'eeg',
                    'data': [channel.__dict__ for channel in eeg_data]
                }, cls=CustomEncoder)))

                if influx:
                    _ = asyncio.create_task(influx.write_to_influx(eeg_data, start_of_epoch, samples_per_epoch, brainflow_input.sampling_rate))
                    # _ = asyncio.create_task(influx.write_raw_to_influx(eeg_data, start_of_epoch, samples_per_epoch, brainflow_input.sampling_rate))

                if lsl:
                    _ = asyncio.create_task(lsl.write_to_lsl(eeg_data, start_of_epoch, samples_per_epoch, brainflow_input.sampling_rate))

        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            pass

    logger.info('Done')


if __name__ == "__main__":
    asyncio.run(run_brainflow())
