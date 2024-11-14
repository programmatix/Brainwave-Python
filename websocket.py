import asyncio

import websockets
import logging
import json
import ssl
from typing import Callable


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsocketHandler:
    def __init__(self, ssl_cert, ssl_key, on_start, on_stop, on_quit, emit_event_callback: Callable[[str, float], None]):
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.server = None
        self.clients = set()
        self.board = None
        self.done = False
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_quit = on_quit
        self.emit_event_callback = emit_event_callback
        self.shutdown_signal = asyncio.Event()

    async def handle_websocket(self, websocket, path = "/"):
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
        if self.ssl_cert and self.ssl_key:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)

            logger.info(f"WebSocket server starting on port {port} using SSL")
            self.server = await websockets.serve(self.handle_websocket, "", port, ssl=ssl_context)
        else:
            logger.info(f"WebSocket server starting on port {port}")
            self.server = await websockets.serve(self.handle_websocket, "", port)
        await self.shutdown_signal.wait()
        await self.server.close()


    def stop(self):
        self.shutdown_signal.set()

    async def process_websocket_message(self, message):
        try:
            msg = json.loads(message)
            logger.info(f"Command received: {message}")
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'message': f"Command '{message}' received"
            }))
            if msg['command'] == 'start':
                logger.info('Starting')
                if 'channels' not in msg:
                    self.on_start(None)
                else:
                    self.on_start(msg['channels'])
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

    def emit_event(self, event_name: str, timestamp: float):
        asyncio.create_task(self.emit_event_callback(event_name, timestamp))
