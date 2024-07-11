import websockets
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsocketHandler:
    def __init__(self, on_start, on_stop, on_quit):
        self.clients = set()
        self.board = None
        self.done = False
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_quit = on_quit

    async def handle_websocket(self, websocket, path):
        logger.info(f"WebSocket connection established with {path}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                logger.info(f"Message from {path}: {message}")
                await self.process_websocket_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection with {path} closed: {e}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection with {path}: {e}")
        finally:
            self.clients.remove(websocket)
            logger.info(f"WebSocket connection with {path} terminated")

    async def start_websocket_server(self, port):
        logger.info(f"WebSocket server starting on port {port}")
        async with websockets.serve(self.handle_websocket, "", port):
            logger.info(f"WebSocket server started on port {port}")
            #await asyncio.Future()

    async def process_websocket_message(self, message):
        try:
            msg = json.loads(message)
            logger.info(f"Command received: {message}")
            if msg['command'] == 'start':
                logger.info('Starting')
                self.on_start()
            elif msg['command'] == 'stop':
                logger.info('Stopping recording')
                self.on_stop()
            elif msg['command'] == 'quit':
                logger.info('Quitting')
                self.on_quit()
            else:
                logger.warning('Unknown command')
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'status': 'success',
                'message': f"Command '{msg['command']}' processed"
            }))
        except Exception as error:
            logger.error(f'Error processing message: {error}')
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'status': 'error',
                'message': f"Command '{message}' failed"
            }))

    async def broadcast_websocket_message(self, message):
        for client in self.clients:
            await client.send(message)
