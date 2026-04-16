import numpy as np

# load each of the three AF images

af4b_973 = np.load('C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\AF\\4B_973_AF.npy')
af4b_974 = np.load('C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\AF\\4B_974_AF.npy')
af4b_975 = np.load('C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\AF\\4B_975_AF.npy')

# find value that covers 99% of pixels so we can clip
print("Channel 0: ")
print(np.percentile(af4b_973[:,:,0], 99))
print(np.percentile(af4b_974[:,:,0], 99))
print(np.percentile(af4b_975[:,:,0], 99))

print("Channel 3: ")
print(np.percentile(af4b_973[:,:,3], 99))
print(np.percentile(af4b_974[:,:,3], 99))
print(np.percentile(af4b_975[:,:,3], 99))


