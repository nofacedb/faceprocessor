#!/usr/bin/python3

"""facerecognition.py is a simple FACEDB face recognition engine"""

import argparse
import asyncio
import json
import ssl
from base64 import b64decode
from io import BytesIO
from re import match

import face_recognition
import numpy as np
import yaml
from PIL import Image
from aiohttp import web, ClientSession


class HTTPServerCFG:
    def __init__(self, cfg: dict):
        self.name = cfg['name']
        self.socket = cfg['socket']
        self.write_timeout_ms = cfg['write_timeout_ms']
        self.read_timeout_ms = cfg['read_timeout_ms']
        self.immed_resp = cfg['immed_resp']
        self.req_max_size = cfg['req_max_size']
        self.key_path = cfg['key_path']
        self.crt_path = cfg['crt_path']


class FaceRecognitionCFG:
    def __init__(self, cfg: dict):
        self.gpu = cfg['gpu']
        self.upsamples = cfg['upsamples']
        self.jitters = cfg['jitters']


class CFG:
    def __init__(self, fcfg: dict):
        self.http_server_cfg = HTTPServerCFG(fcfg['http_server'])
        self.face_recognition_cfg = FaceRecognitionCFG(fcfg['face_recognition'])


class HTTPServer:
    """HTTPServer class handles request for image processing: finding faces and getting facial features."""

    STATUS_BAD_REQUEST = 400
    STATUS_INTERNAL_SERVER_ERROR = 500

    API_V1_GET_FBS = '/api/v1/get_fbs'
    API_V1_GET_FFS = '/api/v1/get_ffs'
    API_V1_PROC_IMG = '/api/v1/proc_img'

    def __init__(self, cfg: CFG):
        self.cfg = cfg
        app = web.Application(client_max_size=self.cfg.http_server_cfg.req_max_size)
        app.add_routes([web.put(HTTPServer.API_V1_GET_FBS, self.get_fbs_handler),
                        web.put(HTTPServer.API_V1_GET_FFS, self.get_ffs_handler),
                        web.put(HTTPServer.API_V1_PROC_IMG, self.proc_img_handler)])
        if self.cfg.http_server_cfg.key_path != '' and self.cfg.http_server_cfg.crt_path != '':
            self.src_addr = 'https://' + self.cfg.http_server_cfg.name
        else:
            self.src_addr = 'http://' + self.cfg.http_server_cfg.name
        self.app = app

    def run(self):
        conn_str = self.cfg.http_server_cfg.socket
        is_ip_port = match(r'(\d)+\.(\d)+\.(\d)+\.(\d)+:(\d)+', conn_str) is not None
        if self.cfg.http_server_cfg.crt_path != '' and self.cfg.http_server_cfg.key_path != '':
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.cfg.http_server_cfg.crt_path,
                                        self.cfg.http_server_cfg.key_path)
        else:
            ssl_context = None
        if is_ip_port:
            conn_data = conn_str.split(':')
            host = conn_data[0]
            port = int(conn_data[1])
            if ssl_context is not None:
                web.run_app(self.app, host=host, port=port, ssl_context=ssl_context)
            else:
                web.run_app(self.app, host=host, port=port)
        else:
            path = conn_str
            if ssl_context is not None:
                web.run_app(self.app, path=path, ssl_context=ssl_context)
            else:
                web.run_app(self.app, path=path)

    RESP_API_V1_PUT_FBS = '/api/v1/put_fbs'

    async def get_fbs_handler(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
            headers = body['headers']
            addr = headers['src_addr']
            req_id = body['id']
            b64_img_buff = body['img_buff']
            img = np.array(Image.open(BytesIO(b64decode(b64_img_buff))))
        except KeyError:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'invalid request data'
            }, status=HTTPServer.STATUS_BAD_REQUEST)

        if self.cfg.http_server_cfg.immed_resp:
            asyncio.ensure_future(self.api_get_fbs_create_resp(addr, req_id, img))
            return web.json_response({'headers': {'src_addr': self.src_addr, 'immed': True}})

        try:
            fbs = await self.get_fbs_from_img(img)
        except Exception:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'unable to get faceboxes from image'
            }, status=HTTPServer.STATUS_INTERNAL_SERVER_ERROR)

        return web.json_response({
            'headers': {'src_addr': self.src_addr, 'immed': False},
            'id': req_id,
            'error': False,
            'boxes': fbs
        })

    async def api_get_fbs_create_resp(self, addr, req_id, img):
        try:
            fbs = await self.get_fbs_from_img(img)
        except Exception:
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PUT_FBS, json={
                    'headers': {'src_addr': self.src_addr, 'immed': False},
                    'id': req_id,
                    'error': True,
                    'error_info': 'unable to get faces from image'
                })
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PUT_FBS, json={
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': False,
                'boxes': fbs
            })

    async def get_fbs_from_img(self, img: np.array) -> list:
        async with self.lock:
            if self.gpu:
                fbs = face_recognition.face_locations(img, self.upsamples, 'cnn')
            else:
                fbs = face_recognition.face_locations(img, self.upsamples, 'hog')
        return fbs

    RESP_API_V1_PUT_FFS = '/api/v1/put_ffs'

    async def get_ffs_handler(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
            headers = body['headers']
            addr = headers['src_addr']
            req_id = body['id']
            b64_img_buff = body['img_buff']
            img = np.array(Image.open(BytesIO(b64decode(b64_img_buff))))
            fbs = body['boxes']
        except KeyError:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'invalid request data'
            }, status=HTTPServer.STATUS_BAD_REQUEST)

        if self.cfg.http_server_cfg.immed_resp:
            asyncio.ensure_future(self.api_get_ffs_create_resp(addr, req_id, img, fbs))
            return web.json_response({'headers': {'src_addr': self.src_addr, 'immed': True}})

        try:
            ffs = await self.get_ffs_from_img(img, fbs)
        except Exception:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'unable to get facial features from image'
            }, status=HTTPServer.STATUS_INTERNAL_SERVER_ERROR)

        return web.json_response({
            'headers': {'src_addr': self.src_addr, 'immed': False},
            'id': req_id,
            'error': False,
            'faces': ffs
        })

    async def api_get_ffs_create_resp(self, addr, req_id, img, fbs):
        try:
            ffs = await self.get_ffs_from_img(img, fbs)
        except Exception:
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PUT_FFS, json={
                    'headers': {'src_addr': self.src_addr, 'immed': False},
                    'id': req_id,
                    'error': True,
                    'error_info': 'unable to get facial features from image'
                })
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PUT_FFS, json={
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': False,
                'faces': ffs
            })

    async def get_ffs_from_img(self, img: np.array, fbs) -> list:
        async with self.lock:
            ffs = face_recognition.face_encodings(img, fbs, self.cfg.face_recognition_cfg.jitters)
        return [{'box': fbs[i], 'features': ffs[i].tolist()} for i in
                range(len(ffs))]

    RESP_API_V1_PROC_IMG = '/api/v1/proc_img'

    async def proc_img_handler(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
            headers = body['headers']
            addr = headers['src_addr']
            req_id = body['id']
            b64_img_buff = body['img_buff']
            img = np.array(Image.open(BytesIO(b64decode(b64_img_buff))))
        except KeyError:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'invalid request data'
            }, status=HTTPServer.STATUS_BAD_REQUEST)

        if self.cfg.http_server_cfg.immed_resp:
            asyncio.ensure_future(self.api_proc_img_create_resp(addr, req_id, img))
            return web.json_response({'headers': {'src_addr': self.src_addr, 'immed': True}})

        try:
            ffs = await self.proc_img(img)
        except Exception:
            return web.json_response({
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': True,
                'error_info': 'unable to process image'
            }, status=HTTPServer.STATUS_INTERNAL_SERVER_ERROR)

        return web.json_response({
            'headers': {'src_addr': self.src_addr, 'immed': False},
            'id': req_id,
            'error': False,
            'faces': ffs
        })

    async def api_proc_img_create_resp(self, addr, req_id, img, fbs):
        try:
            ffs = await self.proc_img(img)
        except Exception:
            async with ClientSession(
                    json_serialize=json.dumps) as session:
                await session.put(addr + HTTPServer.RESP_API_V1_PROC_IMG, json={
                    'headers': {'src_addr': self.src_addr, 'immed': False},
                    'id': req_id,
                    'error': True,
                    'error_info': 'unable to process image'
                })
            return

        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put(addr + HTTPServer.RESP_API_V1_PROC_IMG, json={
                'headers': {'src_addr': self.src_addr, 'immed': False},
                'id': req_id,
                'error': False,
                'faces': ffs
            })

    async def proc_img(self, img: np.array) -> list:
        fbs = await self.get_fbs_from_img(img)
        ffs = await self.get_ffs_from_img(self, img, fbs)
        return ffs


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


def main():
    """main processes cargs and processes input image or runs face recognition server"""
    args = parse_args()
    with open(args.config, 'r') as stream:
        fcfg = yaml.safe_load(stream)
        cfg = CFG(fcfg)

    http_server = HTTPServer(cfg)
    http_server.run()


if __name__ == "__main__":
    main()
