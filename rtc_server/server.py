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
        self.track = track

        # Placeholders
        self._in_img = None
        self._pose_img = None
        self._app_img = None
        self._out_img = None

        self.monkey = Monkey(transform)

        # Continuously update imgs in other threads
        if transform == 'app_transfer':
            worker_1 = Thread(target=self._pose_detection)
            worker_1.setDaemon(True)
            worker_1.start()
            worker_2 = Thread(target=self._appearance_transer)
            worker_2.setDaemon(True)
            worker_2.start()
            self._out_img = self._app_img
        elif transform == 'pose':
            worker = Thread(target=self._pose_detection)
            worker.setDaemon(True)
            worker.start()
            self._out_img = self._pose_img
        else:
            worker = Thread(target=self._std_transform)
            worker.setDaemon(True)
            worker.start()

    async def recv(self):
        frame = await self.track.recv()

        self._in_img = frame.to_ndarray(format='rgb24')

        # Handle initial condition on startup.
        if self._out_img is None:
            return frame

        out_img = self._out_img.astype('uint8')

        # rebuild a VideoFrame, preserving timing information
        # Note that this expects array to be of datatype uint8
        new_frame = VideoFrame.from_ndarray(out_img, format='rgb24')
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def _std_transform(self):
        """
        This is run in a thread only when we are doing no transform.
        It only performs the monkey preprocessing on the image.
        """
        while True:
            self._out_img = self.monkey.preprocess_img(self._in_img)

    def _pose_detection(self):
        """
        Extracts the pose from an input image.
        Used for both pose extraction and appearance transfer.
        """
        while True:
            self._pose_img = self.monkey.draw_pose_from_img(self._in_img)

    def _appearance_transer(self):
        """
        Uses the pose image to perform appearance transfer.
        """
        while True:
            self._out_img = self.monkey.transfer_appearance(self._pose_img)


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

    @pc.on('track')
    def on_track(track):
        log_info('Track %s received', track.kind)

        if track.kind == 'video':
            if params['transform'] == 'original':
                original_video = VideoTransformTrack(track)
                pc.addTrack(original_video)
            elif params['transform'] == 'app_transfer':
                app_transfer_video = VideoTransformTrack(track, transform='app_transfer')
                pc.addTrack(app_transfer_video)

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
    app.router.add_post('/offer_original', offer)
    app.router.add_post('/offer_app_transfer', offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
