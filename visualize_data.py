import numpy as np
import os

from utils import load_data, preview_images

syn_data = np.load(f"/kaggle/working/synthetics/syn_ep{len(os.listdir("/kaggle/working/synthetics"))}.npz", allow_pickle=True)
syn_loader = load_data(syn_data)

preview_images(syn_loader)