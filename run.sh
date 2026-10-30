#!/bin/sh
# Create the virtualenv on first run, then start the notebook at http://127.0.0.1:8888
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv -p 3.13 .venv
  uv pip install -p .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python server.py
