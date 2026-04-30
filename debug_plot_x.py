import numpy as np
import glob
import random
from matplotlib import pyplot as plt

image_path = 'G:/Project_Nectrotic/Data/ProcessingData/RegistrationRound2_crops/NPY/NonNecrotic/BF/*.npy'
all_images = glob.glob(image_path)
bf_path = random.choice(all_images)
af_path = bf_path.replace('BF', 'AF')

label = np.load(bf_path).astype(np.float32) / 255.0
image = np.load(af_path).astype(np.float32)

# match BF dimensions
h, w = label.shape[0], label.shape[1]
image = image[:h, :w, :]

# clip channels to 95th-percentile values
image[:, :, 0] = np.clip(image[:, :, 0], 0, 21776)
image[:, :, 1] = np.clip(image[:, :, 1], 0, 14836)
image[:, :, 2] = np.clip(image[:, :, 2], 0, 6234)
image[:, :, 3] = np.clip(image[:, :, 3], 0, 11038)

# normalize
image = (image - np.mean(image)) / (np.std(image) + 1e-5)

# crop edges
image = image[19:-18, 19:-18, :]
label = label[19:-18, 19:-18, :]

# grab a 256x256 patch from the center
s = 256
cx = image.shape[0] // 2
cy = image.shape[1] // 2
x_patch = image[cx:cx + s, cy:cy + s, :]
y_patch = label[cx:cx + s, cy:cy + s, :]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(-x_patch[:, :, 0], cmap='gray')
axes[0].set_title(f'x channel 0 (AF, negated for display)\n{bf_path.split(chr(92))[-1]}')
axes[0].axis('off')

axes[1].imshow(y_patch)
axes[1].set_title('y (BF label)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f'x shape: {x_patch.shape}, min: {x_patch[:,:,0].min():.3f}, max: {x_patch[:,:,0].max():.3f}')
print(f'y shape: {y_patch.shape}, min: {y_patch.min():.3f}, max: {y_patch.max():.3f}')
