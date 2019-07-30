import re
import subprocess

from builtin_parser import *;

def preprocess_builtins(builtins_epi_path):
    preprocessed = subprocess.check_output(["cpp", "-P", "-o-", \
            builtins_epi_path])
    RE = r"EPI_BUILTIN\s*\(\s*([a-z0-9_]+)\s*,\s*\"([^\"]+)\"\s*,\s*\"[^\"]*\"\s*\)"
    for m in re.finditer(RE, preprocessed):
        yield (m.group(1), m.group(2))
