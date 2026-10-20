#!/bin/bash
# Reference solution. Only touches source files, never the visible tests.
set -euo pipefail

cd /app
cp /solution/proration.py /app/domain/rules/proration.py
cp /solution/settings.py /app/config/settings.py

python3 -m pytest tests -q
