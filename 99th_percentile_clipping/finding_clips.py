import numpy as np
import glob
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

image_path = 'G:/Project_Nectrotic/Data/ProcessingData/RegistrationRound2_crops/NPY/NonNecrotic/AF/*.npy'
all_images = glob.glob(image_path)

sample_images = [f for f in all_images if os.path.basename(f)[0] in ['1', '2']]

percentile_95 = [[], [], [], []]
percentile_99 = [[], [], [], []]

for i in range(sample_images.__len__()):
    image = np.load(sample_images[i])
    for j in range(4):
        percentile_95[j].append(np.percentile(image[:, :, j], 95))
        percentile_99[j].append(np.percentile(image[:, :, j], 99))
    
print("Channel 0 95th percentile mean: ", np.mean(percentile_95[0]))
print("Channel 0 99th percentile mean: ", np.mean(percentile_99[0]))
print("Channel 1 95th percentile mean: ", np.mean(percentile_95[1]))
print("Channel 1 99th percentile mean: ", np.mean(percentile_99[1]))
print("Channel 2 95th percentile mean: ", np.mean(percentile_95[2]))
print("Channel 2 99th percentile mean: ", np.mean(percentile_99[2]))
print("Channel 3 95th percentile mean: ", np.mean(percentile_95[3]))
print("Channel 3 99th percentile mean: ", np.mean(percentile_99[3]))
