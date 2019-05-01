#!/usr/bin/python3

"""facerecognition.py is a simple FACEDB face recognition engine"""

import argparse
import asyncio
from re import match

from io import BytesIO
import face_recognition
import numpy as np
from PIL import Image
from aiohttp import web

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
    parser.add_argument('--reqmaxsize', type=int, default=1024**2,
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

    def __init__(self, gpu: bool, upsamples: int, jitters: int):
        self.gpu = gpu
        self.upsamples = upsamples
        self.jitters = jitters
        self.lock = asyncio.Lock()

    async def handle(self, req: web.Request) -> web.Response:
        """handle handles requests to process image"""
        img_buff = await req.read()
        img = Image.open(BytesIO(img_buff))
        img = np.array(img)
        async with self.lock:
            if self.gpu:
                face_locations = face_recognition.face_locations(img, self.upsamples, 'cnn')
            else:
                face_locations = face_recognition.face_locations(img, self.upsamples, 'hog')
            face_encodings = face_recognition.face_encodings(img, face_locations, self.jitters)

        face_locations = [np.array(fl).astype(np.uint64) for fl in face_locations]
        data_buff = BytesIO()
        for i in range(len(face_locations)):
            data_buff.write(face_locations[i].astype(np.uint64).tobytes())
            data_buff.write(face_encodings[i].astype(np.float64).tobytes())
        data_buff.seek(0)
        resp = web.Response(body=data_buff.read())
        data_buff.close()
        return resp


def server_face_recognition(args) -> int:
    """server_face_recognition starts asynchronous face recognition server"""
    conn_str = args.socket
    is_ip_port = match(r'(\d)+\.(\d)+\.(\d)+\.(\d)+:(\d)+', conn_str) is not None
    app_ctx = AppCtx(args.gpu, args.upsamples, args.jitters)
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
