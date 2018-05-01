#!/bin/bash

# Train a sentiment_analysis model (default: conv model), with a user
# specified seed
python paddle/train.py -s $1
