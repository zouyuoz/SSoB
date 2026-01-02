import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
import time
import datetime
import os
from tqdm import tqdm

device = torch.device("cuda")
# device = torch.device("cpu")
device