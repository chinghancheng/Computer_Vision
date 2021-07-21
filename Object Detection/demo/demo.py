import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd

net = build_ssd('test', 300, 21)    # initialize SSD

print(net)

net.load_weights('ssd300_mAP_77.43_v2.pth')

