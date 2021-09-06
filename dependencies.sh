#!/bin/sh
conda create -n emotion-detect python=3.8 -y
conda activate emotion-detect
pip install -r requirements.txt
pip install git+https://github.com/rcmalli/keras-vggface.git
