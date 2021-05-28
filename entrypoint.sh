#!/usr/bin/env bash

exec gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py main:app
#exec uvicorn --host 0.0.0.0 --port 80 main:app