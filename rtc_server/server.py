import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

from rtc_server.monkey import Monkey

ROOT = os.path.dirname(__file__)

logger = logging.getLogger('pc')
pcs = set()


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform=None):
        super().__init__()  # don't forget this!
        self.counter = 0
        self.track = track
        if transform:
            self.monkey = Monkey()
            self.worker = Thread(target=self.draw_pose_img)
            self.worker.setDaemon(True)
            self.worker.start()

        # Placeholders
        self.in_img = None
        self.pose_img = None

    async def recv(self):
        frame = await self.track.recv()
        self.counter += 1

        self.in_img = frame.to_ndarray(format='bgr24')

        # Hanlde initial condition on startup.
        if self.pose_img is None:
            return frame

        out_img = (self.pose_img*256).astype('uint8')

        # rebuild a VideoFrame, preserving timing information
        # Note that this expects array to be of datatype uint8
        new_frame = VideoFrame.from_ndarray(out_img, format='bgr24')
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def draw_pose_img(self):
        """
        Detect pose and draw inside a seperate thread.
        """
        while True:
            self.pose_img = self.monkey.draw_pose_from_img(self.in_img)

async def index(request):
    content = open(os.path.join(ROOT, 'index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, 'client.js'), 'r').read()
    return web.Response(content_type='application/javascript', text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection()
    pc_id = 'PeerConnection(%s)' % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + ' ' + msg, *args)

    log_info('Created for %s', request.remote)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            if isinstance(message, str) and message.startswith('ping'):
                channel.send('pong' + message[4:])

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        log_info('ICE connection state is %s', pc.iceConnectionState)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        log_info('Track %s received', track.kind)

        if track.kind == 'video':
            transformed_video = VideoTransformTrack(track, transform=params['video_transform'])
            pc.addTrack(transformed_video)

        @track.on('ended')
        async def on_ended():
            log_info('Track %s ended', track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRTC video / data-channels demo')
    parser.add_argument('--cert-file', help='SSL certificate file (for HTTPS)')
    parser.add_argument('--key-file', help='SSL key file (for HTTPS)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for HTTP server (default: 8080)')
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
