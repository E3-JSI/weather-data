#!/usr/bin/env python
import ecmwfapi
import re
import os
import argparse

parser = argparse.ArgumentParser(description='Run MARS request.')
parser.add_argument('infile', nargs='?', default='-', type=argparse.FileType('r'),
                    help='file containing a MARS request or STDIN otherwise')

args = parser.parse_args()
req = args.infile.read()

if "WEBMARS_TARGET" in os.environ:
    target = os.environ["WEBMARS_TARGET"]
else:
    m = re.search(r'\btar(g(e(t)?)?)?\s*=\s*([^\'",\s]+|"[^"]*"|\'[^\']*\')', req, re.I|re.M)
    if m is None:
        raise Exception("Cannot extract target")

    target=m.group(4)
    if target is None:
        raise Exception("Cannot extract target")

if target[0] == target[-1]:
    if target[0] in ['"', "'"]:
        target = target[1:-1]

c = ecmwfapi.ECMWFService('mars')
c.execute(req, target)
