#!/bin/bash

python conddecoder.py --decoder_hidden_dim 1024 --tags hidden1024

python conddecoder.py --decoder_hidden_dim 512 --tags hidden512

python conddecoder.py --decoder_hidden_dim 256 --tags hidden256
