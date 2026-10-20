#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app
cp /solution/engine.py /app/engine.py

python3 -m pytest tests -q
