#!/bin/bash

python annotation.py
python process.py
python merge.py
python clean.py
python normalization.py
