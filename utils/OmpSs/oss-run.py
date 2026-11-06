#!/usr/bin/env python3
# Run an OmpSs-2 test program. nOS-V only removes its per-process shared memory
# segment on a clean shutdown, so remove it in case the program crashed.
import os
import subprocess
import sys

p = subprocess.Popen(sys.argv[1:])
ret = p.wait()

try:
    os.unlink("/dev/shm/nosv-u{}-p{}".format(os.geteuid(), p.pid))
except OSError:
    pass

sys.exit(128 - ret if ret < 0 else ret)
