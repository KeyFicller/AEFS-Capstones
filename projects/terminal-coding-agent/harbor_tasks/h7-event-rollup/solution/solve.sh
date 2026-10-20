#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app
cp /solution/sessionize.py /app/sessionize.py

python3 -m pytest tests -q
