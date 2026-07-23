#!/usr/bin/env python3
import sys, re

msg = sys.stdin.read()
msg = re.sub(r'(?im)^Co-Authored-By:\s*Claude.*\n?', '', msg)
sys.stdout.write(msg.rstrip('\n') + '\n')
