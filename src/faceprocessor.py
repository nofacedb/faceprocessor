#!/usr/bin/python3

"""facerecognition.py is a simple FACEDB face recognition engine"""

import argparse
import asyncio
import json
from base64 import b64decode
from io import BytesIO
from re import match

import face_recognition
import numpy as np
from PIL import Image
from aiohttp import web, ClientSession

U_INT64_SIZE = np.dtype(np.uint64).itemsize
FLOAT64_SIZE = np.dtype(np.float64).itemsize

DESC_STR = r"""FaceRecognition is a simple script, that finds all faces in image
and returns their coordinates and features vectors.
"""


def parse_args():
    """parse_args parses all command line arguments"""
    parser = argparse.ArgumentParser(prog='FaceRecognition',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=DESC_STR)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='path to yaml config file')
    parser.add_argument('-s', '--socket', type=str, default='',
                        help='IP:PORT or path to UNIX-Socket')
    parser.add_argument('--immedresp', action='store_true',
                        help='specifies, if server should return answer immediately')
    parser.add_argument('--reqmaxsize', type=int, default=1024 ** 2,
                        help='client request max size in bytes (default: %(default)s))')
    parser.add_argument('-i', '--input', type=str, default='',
                        help='path to input image')
    parser.add_argument('-g', '--gpu', action='store_true',
                        help='use GPU')
    parser.add_argument('-u', '--upsamples', type=int, default=1,
                        help='number of upsamples (bigger ->'
                             ' smaller faces could be found (default: %(default)s))')
    parser.add_argument('-j', '--jitters', type=int, default=1,
                        help='number of re-samples (bigger -> better quality '
                             'of features vectors, but slower (default: %(default)s))')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='path to output image')

    args = parser.parse_args()

    return args


def file_face_recognition(args):
    """file_face_recognition processes only one input image"""
    try:
        img = face_recognition.load_image_file(args.input)

        if args.gpu:
            face_locations = face_recognition.face_locations(img, args.upsamples, 'cnn')
        else:
            face_locations = face_recognition.face_locations(img, args.upsamples, 'hog')

        face_encodings = face_recognition.face_encodings(img, face_locations, args.jitters)
        face_locations = [np.array(fl).astype(np.uint64) for fl in face_locations]

        with open(args.output, 'wb') as out:
            for i in range(len(face_locations)):
                face_locations[i].astype(np.uint64).tofile(out)
                face_encodings[i].astype(np.float64).tofile(out)
        return 0
    except IOError:
        return -1


class AppCtx:
    """AppCtx class is used for scheduling face recognition tasks"""

    def __init__(self, immed_resp: bool, gpu: bool, upsamples: int, jitters: int):
        self.immed_resp = immed_resp
        self.gpu = gpu
        self.upsamples = upsamples
        self.jitters = jitters
        self.lock = asyncio.Lock()
        self.tasks = []

    async def handle(self, req: web.Request) -> web.Response:
        """handle handles requests to process image"""
        body = await req.json()
        headers = body.get('headers')
        id = body.get('id')
        b64_img_buff = body.get('img_buff')
        img_buff = b64decode(b64_img_buff)

        if self.immed_resp:
            asyncio.ensure_future(self.create_response(headers, id, img_buff))
            return web.json_response({'headers': {'src_addr': '', 'immed': True}})

        faces = await self.process_img(img_buff)
        resp = {
            'headers': {'src_addr': '', 'immed': False},
            'id': id,
            'faces': faces
        }
        return web.json_response(resp)

    async def create_response(self, headers, id, img_buff):
        faces = await self.process_img(img_buff)
        async with ClientSession(
                json_serialize=json.dumps) as session:
            await session.put('http://127.0.0.1:10000' + '/api/v1/put_features', json={
                'headers': {'src_addr': '', 'immed': False},
                'id': id,
                'faces': faces
            })
        print('process request for id', id)

    async def process_img(self, img_buff):
        img = Image.open(BytesIO(img_buff))
        img = np.array(img)
        async with self.lock:
            if self.gpu:
                face_locations = face_recognition.face_locations(img, self.upsamples, 'cnn')
            else:
                face_locations = face_recognition.face_locations(img, self.upsamples, 'hog')
            face_encodings = face_recognition.face_encodings(img, face_locations, self.jitters)
        return [{'box': face_locations[i], 'features': face_encodings[i].tolist()} for i in
                range(len(face_encodings))]


def server_face_recognition(args) -> int:
    """server_face_recognition starts asynchronous face recognition server"""
    conn_str = args.socket
    is_ip_port = match(r'(\d)+\.(\d)+\.(\d)+\.(\d)+:(\d)+', conn_str) is not None
    app_ctx = AppCtx(args.immedresp, args.gpu, args.upsamples, args.jitters)
    app = web.Application(client_max_size=args.reqmaxsize)
    app.add_routes([web.put('/', app_ctx.handle)])

    if is_ip_port:
        conn_data = conn_str.split(':')
        host = conn_data[0]
        port = int(conn_data[1])
        web.run_app(app, host=host, port=port)
    else:
        path = conn_str
        web.run_app(app, path=path)

    return 0


def main():
    """main processes cargs and processes input image or runs face recognition server"""
    args = parse_args()
    if args.socket == '':
        return file_face_recognition(args)
    return server_face_recognition(args)


if __name__ == "__main__":
    main()
