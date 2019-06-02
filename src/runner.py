#!/usr/bin/python3

import argparse

import subprocess


def parse_args():
    parser = argparse.ArgumentParser(prog='Runner',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='path to yaml config file')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    while(True):
        print('(re)started faceprocessor')
        try:
            process = subprocess.Popen(['./faceprocessor.py', '-c', args.config])
            process.wait()
        except Exception:
            print('faceprocessor died, need to restart it.')


if __name__ == "__main__":
    main()
