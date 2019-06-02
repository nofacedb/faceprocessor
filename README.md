# faceprocessor

**nofacedb/faceprocessor** is a face recognition engine for **NoFaceDB** program complex.

## Tech

**faceprocessor** uses a number of open source projects to work properly:
- [face_recognition](https://github.com/ageitgey/face_recognition) - simple but cool API for processing faces on images;
- [AIOHTTP](https://aiohttp.readthedocs.io/en/stable/) - asynchronous HTTP library;

## Installation

**faceprocessor** requires [Python](https://www.python.org/) v3.6+ to run.

Get **faceprocessor** (and other microservices), install the dependencies from requirements.txt, and now You are ready to find faces!

```sh
$ git clone https://github.com/nofacedb/faceprocessor
$ cd faceprocessor
$ pip install -r requirements.txt
```
## HowTo
**faceprocessor** consists of two main scripts: `src/faceprocessor.py` and `src/runner.py`. First is a complete server + facerecognizer and second is it's suprevisor. Because of `faceprocessor.py` can't clean GPU memory after it processes image (and executing it for every new image is too slow), it processes images until it fails, and ther `runner.py` restarts it.

## Many thanks to:

- Igor Vishnyakov and Mikhail Pinchukov - my scientific directors;
