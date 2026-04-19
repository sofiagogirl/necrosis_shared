import numpy as np

# ======================= config =================================
data_dir = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\AF\\'
samples = ['4B_973', '4B_974', '4B_975']
channels = [0, 3]
percentile = 99

# ======================= load data ==============================
af_images = {s: np.load(data_dir + f'{s}_AF.npy') for s in samples}

# ======================= percentile analysis ====================
for ch in channels:
    print(f"Channel {ch}:")
    for s, img in af_images.items():
        print(f"  {s}: {np.percentile(img[:, :, ch], percentile):.2f}")
    print()