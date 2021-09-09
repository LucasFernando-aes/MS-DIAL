#!/bin/bash

echo "python main.py --name=digits --gamma=0.1 --batch_size=128 $@"
python main.py --name=digits --gamma=0.1 --batch_size=128 $@
