#!/bin/bash

echo "python main.py --name=office31 --gamma=0.5 --batch_size=32 $@"
python main.py --name=office31 --gamma=0.5 --batch_size=32 $@
