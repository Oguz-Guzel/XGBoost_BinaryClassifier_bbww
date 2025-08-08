#!/bin/bash
source /afs/cern.ch/work/a/aguzel/private/bamboo_105/pytorch_env/bin/activate
cd /afs/cern.ch/work/a/aguzel/private/bamboo_105/HHtoWWbb_Run3/XGBoost
python XGBoost.py --output_dir $1