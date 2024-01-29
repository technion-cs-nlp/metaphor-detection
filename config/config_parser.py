# parser --------------------------------------------------------------------------
import yaml
from box import Box
import os

import platform
global training_args

curr_path = os.getcwd()

if platform.system() == 'Darwin':
    with open('{}/config/config.yaml'.format(curr_path)) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
else:
    with open('{}/config//config.yaml'.format(curr_path)) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
