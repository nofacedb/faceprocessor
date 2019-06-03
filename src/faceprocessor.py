#!/usr/bin/python3

"""facerecognition.py is a simple FACEDB face recognition engine"""

import argparse
import asyncio
import json
import logging
import ssl
import sys
from asyncio import Lock
from base64 import b64decode
from io import BytesIO

import face_recognition
import numpy as np
import yaml
from PIL import Image
from aiohttp import web, ClientSession


class HTTPServerCFG:
    def __init__(self, cfg: dict):
        self.addr = cfg['addr']
        self.port = cfg['port']
        self.write_timeout_ms = cfg['write_timeout_ms']
        self.read_timeout_ms = cfg['read_timeout_ms']
        self.req_max_size = cfg['req_max_size']
        self.key_path = cfg['key_path']
        self.crt_path = cfg['crt_path']


class FaceRecognitionCFG:
    def __init__(self, cfg: dict):
        self.gpu = cfg['gpu']
        self.upsamples = cfg['upsamples']
        self.jitters = cfg['jitters']
        self.max_width = cfg['max_width']
        self.max_height = cfg['max_height']
        self.timeout_ms = cfg['timeout_ms']


class LoggerCFG:
    def __init__(self, cfg: dict):
        self.output = cfg['output']


class CFG:
    def __init__(self, fcfg: dict):
        self.http_server_cfg = HTTPServerCFG(fcfg['http_server'])
        self.face_recognition_cfg = FaceRecognitionCFG(fcfg['face_recognition'])
        self.logger_cfg = LoggerCFG(fcfg['logger'])


class HTTPServer:
    """HTTPServer class handles requests for image processing: finding faces and getting facial features."""

    INVALID_REQUEST_METHOD_CODE = -1
    CORRUPTED_BODY_CODE = -2
    UNABLE_TO_ENQUEUE = -3
    UNABLE_TO_SEND = -4
    INTERNAL_SERVER_ERROR = -5

    STATUS_BAD_REQUEST = 400
    STATUS_INTERNAL_SERVER_ERROR = 500

    API_BASE = '/api/v1'
    API_GET_FACEBOXES = API_BASE + '/get_faceboxes'
    API_GET_FACIAL_FEATURES_VECTORS = API_BASE + '/get_facial_features_vectors'
    API_PROCESS_IMAGE = API_BASE + '/process_image'

    def __init__(self, cfg: CFG, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        app = web.Application(client_max_size=self.cfg.http_server_cfg.req_max_size)
        app.add_routes([web.put(HTTPServer.API_GET_FACEBOXES, self.get_faceboxes_handler),
                        web.put(HTTPServer.API_GET_FACIAL_FEATURES_VECTORS, self.get_facial_features_vectors),
                        web.put(HTTPServer.API_PROCESS_IMAGE, self.process_image_handler)])
        self.app = app
        if self.cfg.http_server_cfg.key_path != '' and self.cfg.http_server_cfg.crt_path != '':
            self.src_addr = 'https://' + self.cfg.http_server_cfg.addr + ':' + str(self.cfg.http_server_cfg.port)
        else:
            self.src_addr = 'http://' + self.cfg.http_server_cfg.addr + ':' + str(self.cfg.http_server_cfg.port)
        self.lock = Lock()

    def run(self):
        if self.cfg.http_server_cfg.crt_path != '' and self.cfg.http_server_cfg.key_path != '':
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.cfg.http_server_cfg.crt_path,
                                        self.cfg.http_server_cfg.key_path)
        else:
            ssl_context = None
        if ssl_context is not None:
            web.run_app(self.app, host=self.cfg.http_server_cfg.addr,
                        port=self.cfg.http_server_cfg.port, ssl_context=ssl_context)
        else:
            web.run_app(self.app, host=self.cfg.http_server_cfg.addr,
                        port=self.cfg.http_server_cfg.port)

    RESP_API_V1_PUT_FACEBOXES = '/api/v1/put_faceboxes'

    async def get_faceboxes_handler(self, req: web.Request) -> web.Response:
        self.logger.info('got request on %s' % HTTPServer.API_GET_FACEBOXES)
        req_uuid = ''
        try:
            body = await req.json()
            header = body['header']
            addr = header['src_addr']
            req_uuid = header['uuid']
            img_buff = body['img_buff']
            img, coef = self.scale_image(Image.open(BytesIO(b64decode(img_buff))))
            img = np.array(img)
        except Exception:
            self.logger.exception('unable to process request')
            return web.json_response({
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'error_data': {
                    'error_code': HTTPServer.CORRUPTED_BODY_CODE,
                    'error_info': 'corrupted request body',
                    'error_text': 'unable to read request body'
                }
            }, status=HTTPServer.STATUS_BAD_REQUEST)
        self.logger.debug('SrcAddr: "%s", UUID: "%s"', addr, req_uuid)

        asyncio.ensure_future(self.get_faceboxes_create_resp(addr, req_uuid, img, coef))
        return web.json_response({'header': {'src_addr': self.src_addr, 'uuid': req_uuid}})

    async def get_faceboxes_create_resp(self, addr, req_uuid, img, coef):
        try:
            self.logger.debug('getting faceboxes for image "%s"' % req_uuid)
            faceboxes = await self.get_faceboxes_from_image(img)
            for i in range(len(faceboxes)):
                facebox = faceboxes[i]
                facebox = (int(facebox[0] * coef),
                           int(facebox[1] * coef),
                           int(facebox[2] * coef),
                           int(facebox[3] * coef))
                faceboxes[i] = facebox
            self.logger.debug('successfully processed image "%s"' % req_uuid)
        except Exception:
            self.logger.exception('unable to process image "%s"' % req_uuid)
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACEBOXES, json={
                    '': {'src_addr': self.src_addr, 'uuid': req_uuid},
                    'error_data': {
                        'error_code': HTTPServer.INTERNAL_SERVER_ERROR,
                        'error_info': 'unable to get faceboxes',
                        'error_text': 'unable to get faceboxes'
                    }
                })
            exit(1)
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACEBOXES, json={
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'faceboxes': faceboxes
            })
        self.logger.debug('successfully sent faceboxes data for image "%s" to "%s"' % (req_uuid, addr))

    async def get_faceboxes_from_image(self, img: np.array) -> list:
        if self.cfg.face_recognition_cfg.gpu:
            faceboxes = face_recognition.face_locations(img, self.cfg.face_recognition_cfg.upsamples, 'cnn')
        else:
            faceboxes = face_recognition.face_locations(img, self.cfg.face_recognition_cfg.upsamples, 'hog')
        return faceboxes

    RESP_API_V1_PUT_FACES_DATA = '/api/v1/put_faces_data'

    async def get_facial_features_vectors(self, req: web.Request) -> web.Response:
        self.logger.info('got request on %s' % HTTPServer.API_GET_FACIAL_FEATURES_VECTORS)
        req_uuid = ''
        try:
            body = await req.json()
            header = body['header']
            addr = header['src_addr']
            req_uuid = header['uuid']
            img_buff = body['img_buff']
            img, coef = self.scale_image(Image.open(BytesIO(b64decode(img_buff))))
            img = np.array(img)
            faceboxes = body['faceboxes']
        except Exception:
            self.logger.exception("unable to process request")
            return web.json_response({
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'error_data': {
                    'error_code': HTTPServer.CORRUPTED_BODY_CODE,
                    'error_info': 'corrupted request body',
                    'error_text': 'unable to read request body'
                }
            }, status=HTTPServer.STATUS_BAD_REQUEST)
        self.logger.debug('SrcAddr: "%s", UUID: "%s"', addr, req_uuid)

        asyncio.ensure_future(self.api_get_facial_features_vectors_create_resp(addr, req_uuid, img, coef, faceboxes))
        return web.json_response({'header': {'src_addr': self.src_addr, 'immed': True}})

    async def api_get_facial_features_vectors_create_resp(self, addr, req_uuid, img, coef, faceboxes):
        try:
            self.logger.debug('getting facial features vectors for image "%s"' % req_uuid)
            for i in range(len(faceboxes)):
                facebox = faceboxes[i]
                facebox = (int(facebox[0] / coef),
                           int(facebox[1] / coef),
                           int(facebox[2] / coef),
                           int(facebox[3] / coef))
                faceboxes[i] = facebox
            ffs = await self.get_facial_features_vectors_from_image(img, faceboxes)
            for ff in ffs:
                facebox = ff['facebox']
                facebox = (int(facebox[0] * coef),
                           int(facebox[1] * coef),
                           int(facebox[2] * coef),
                           int(facebox[3] * coef))
                ff['facebox'] = facebox
            self.logger.debug('successfully processed image "%s"' % req_uuid)
        except Exception:
            self.logger.exception('unable to process image "%s"' % req_uuid)
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACES_DATA, json={
                    'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                    'error_data': {
                        'error_code': HTTPServer.INTERNAL_SERVER_ERROR,
                        'error_info': 'unable to get facial features vectors',
                        'error_text': 'unable to get facial features vectors'
                    }
                })
            exit(1)
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACES_DATA, json={
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'faces_data': ffs
            })
        self.logger.debug('successfully sent faces data for image "%s" to "%s"' % (req_uuid, addr))

    async def get_facial_features_vectors_from_image(self, img: np.array, fbs) -> list:
        ffs = face_recognition.face_encodings(img, fbs, self.cfg.face_recognition_cfg.jitters)
        return [{'facebox': fbs[i], 'facial_features_vector': ffs[i].tolist()} for i in
                range(len(ffs))]

    async def process_image_handler(self, req: web.Request) -> web.Response:
        self.logger.info('got request on %s' % HTTPServer.API_PROCESS_IMAGE)
        req_uuid = ''
        try:
            body = await req.json()
            header = body['header']
            addr = header['src_addr']
            req_uuid = header['uuid']
            img_buff = body['img_buff']
            img, coef = self.scale_image(Image.open(BytesIO(b64decode(img_buff))))
            img = np.array(img)
        except Exception:
            self.logger.exception("unable to process request")
            return web.json_response({
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'error_data': {
                    'error_code': HTTPServer.CORRUPTED_BODY_CODE,
                    'error_info': 'corrupted request body',
                    'error_text': 'unable to read request body'
                }
            }, status=HTTPServer.STATUS_BAD_REQUEST)
        self.logger.debug('SrcAddr: "%s", UUID: "%s"', addr, req_uuid)

        asyncio.ensure_future(self.api_process_image_create_resp(addr, req_uuid, img, coef))
        return web.json_response({'header': {'src_addr': self.src_addr, 'uuid': req_uuid}})

    async def api_process_image_create_resp(self, addr, req_uuid, img, coef):
        try:
            self.logger.debug('processing image "%s"' % req_uuid)
            ffs = await self.process_image(img)
            for ff in ffs:
                facebox = ff['facebox']
                facebox = (int(facebox[0] * coef),
                           int(facebox[1] * coef),
                           int(facebox[2] * coef),
                           int(facebox[3] * coef))
                ff['facebox'] = facebox
            self.logger.debug('successfully processed image "%s"' % req_uuid)
        except Exception:
            self.logger.exception('unable to process image "%s"' % req_uuid)
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACES_DATA, json={
                    'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                    'error_data': {
                        'error_code': HTTPServer.INTERNAL_SERVER_ERROR,
                        'error_info': 'unable to process image',
                        'error_text': 'unable to process image'
                    }
                })
            exit(1)
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PUT_FACES_DATA, json={
                'header': {'src_addr': self.src_addr, 'uuid': req_uuid},
                'faces_data': ffs
            })
        self.logger.debug('successfully sent faces data for image "%s" to "%s"' % (req_uuid, addr))

    async def process_image(self, img: np.array) -> list:
        fbs = await self.get_faceboxes_from_image(img)
        ffs = await self.get_facial_features_vectors_from_image(img, fbs)
        return ffs

    def scale_image(self, img: Image.Image) -> (Image.Image, float):
        scale_coef_width = img.width / self.cfg.face_recognition_cfg.max_width
        scale_coef_height = img.height / self.cfg.face_recognition_cfg.max_height
        if (scale_coef_width >= 1.0) or (scale_coef_height >= 1.0):
            if scale_coef_width > scale_coef_height:
                coef = scale_coef_width
            else:
                coef = scale_coef_height
        else:
            coef = 1.0
        size = (int(img.width / coef), int(img.height / coef))
        img = img.resize(size)
        return img, coef


DESC_STR = r"""FaceRecognition is a simple script, that finds all faces in image
and returns their coordinates and features vectors.
"""


def parse_args():
    parser = argparse.ArgumentParser(prog='FaceRecognition',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=DESC_STR)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='path to yaml config file')

    args = parser.parse_args()

    return args


def create_logger(cfg: LoggerCFG) -> logging.Logger:
    if cfg.output == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
    elif cfg.output == 'stderr':
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(cfg.output)
    handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger("fr")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def main():
    args = parse_args()
    with open(args.config, 'r') as stream:
        fcfg = yaml.safe_load(stream)
        cfg = CFG(fcfg)

    logger = create_logger(cfg.logger_cfg)

    logger.debug('initializing HTTP SERVER...')
    http_server = HTTPServer(cfg, logger)
    logger.debug('HTTP SERVER was sucessfully initialized')
    http_server.run()


if __name__ == "__main__":
    main()
