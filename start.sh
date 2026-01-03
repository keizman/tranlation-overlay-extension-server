#!/bin/bash

pkill -f 9110
nohup uvicorn main:app --host 127.0.0.1 --port 9110 > running.log 2>&1 &