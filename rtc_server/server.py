import argparse
from datetime import datetime
import asyncio
import json
import logging
import os
import ssl
import uuid
from threading import Thread

from aiohttp import web
from av import VideoFrame

from aiortc import (RTCPeerConnection, RTCSessionDescription,
                    VideoStreamTrack)
from aiortc.contrib.media import MediaBlackhole

from rtc_server.monkey import Monkey

ROOT = os.path.dirname(__file__)

logger = logging.getLogger('pc')
pcs = set()


class ImageTransformer:
    """
    Runs all the transformations in threads that VideoTransformTrack
    can then pull required data from.
    Makes things more efficient as GPU doesn't have to recalculate
    anything.
    """
    def __init__(self):
        # Placeholders
        self.in_img = None
        self.preprocessed_img = None
        self.pose_img = None
        self.app_img = None

        self.monkey = Monkey()

    def start_threads(self):
        """
        Run the transformations in seperate threds to increase
        efficiency.
        """
        preprocess_worker = Thread(target=self._std_transform)
        pose_worker = Thread(target=self._pose_detection)
        app_worker = Thread(target=self._appearance_transer)

        workers = [preprocess_worker, pose_worker] #, app_worker]
        for worker in workers:
            worker.setDaemon(True)
            worker.start()

    def _std_transform(self):
        """
        This is run in a thread only when we are doing no transform.
        It only performs the monkey preprocessing on the image.
        """
        while True:
            self.preprocessed_img = self.monkey.preprocess_img(self.in_img)
            # TODO: sort out
            # WebRTC wants in range 0-256
            # if self.in_img is not None:
            #    self.preprocessed_img *= 256

    def _pose_detection(self):
        """
        Extracts the pose from an input image.
        Used for both pose extraction and appearance transfer.
        """
        while True:
            #start_time = datetime.now()
            input_img = None
            if self.in_img is not None:
                input_img = self.in_img / 256
            pose_img = self.monkey.draw_pose_from_img(input_img)

            # WebRTC wants in range 0-256
            if pose_img is not None:
                self.pose_img = pose_img * 256
            #delay = (datetime.now() - start_time).total_seconds()
            #print("{:.1f} frames per second".format(1/delay))

    def _appearance_transer(self):
        """
        Uses the pose image to perform appearance transfer.
        """
        while True:
            input_pose = None
            if self.pose_img is not None:
                input_pose = self.pose_img / 256
            app_img = self.monkey.transfer_appearance(input_pose)

            # WebRTC wants in range 0-256
            if app_img is not None:
                self.app_img = app_img * 256


# Start the threads going as soon as possible in the global scope
IMAGE_TRANSFORMER = ImageTransformer()
IMAGE_TRANSFORMER.start_threads()


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        # Only have one track (the standard one) setting the input image
        if self.transform not in ['appearance', 'pose']:
            IMAGE_TRANSFORMER.in_img = frame.to_ndarray(format='rgb24')

        # Handle initial condition on startup.
        if IMAGE_TRANSFORMER.preprocessed_img is None or \
           IMAGE_TRANSFORMER.pose_img is None: #or \
            return frame
           #IMAGE_TRANSFORMER.app_img is None:
           # return frame

        #if self.transform == 'appearance':
        #    out_img = IMAGE_TRANSFORMER.app_img.astype('uint8')
        if self.transform == 'pose':
            out_img = IMAGE_TRANSFORMER.pose_img.astype('uint8')
        else:
            out_img = IMAGE_TRANSFORMER.preprocessed_img.astype('uint8')

        # rebuild a VideoFrame, preserving timing information
        # Note that this expects array to be of datatype uint8
        new_frame = VideoFrame.from_ndarray(out_img, format='rgb24')
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


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
    # TODO: Check if 'recorder' can be removed
    recorder = MediaBlackhole()

    @pc.on('track')
    def on_track(track):
        log_info('Track %s received', track.kind)

        if track.kind == 'video':
            transform = params['transform']
            output = VideoTransformTrack(track, transform=transform)
            pc.addTrack(output)

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
    parser = argparse.ArgumentParser()
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
    app.router.add_post('/offer_pose', offer)
    app.router.add_post('/offer_appearance', offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
