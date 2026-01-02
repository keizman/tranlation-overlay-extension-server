#!/bin/bash

pkill -f 9110
nohup uvicorn main:app --host 142.171.1.88 --port 9110 > running.log 2>&1 &