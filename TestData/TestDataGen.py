import numpy as np
import os
import random
from matplotlib.pyplot import imsave

imgs = np.load("../MajorData/09AZ_space__binary_images.npy")
labels = np.load("../MajorData/09AZ_space__binary_labels.npy")


os.system("rm ./*.jpg")

for images in range(100):
    rc = random.randint(10,30)
    cc = random.randint(10,30)
    for _ in range(cc):
        row = np.concatenate([imgs[random.randint(0,imgs.shape[0]-1)] for _ in range(rc)],axis = 1)
        if _ == 0:
            test = np.concatenate([row,np.zeros((row.shape[0]*2,row.shape[1],row.shape[2]))],axis=0)
        else:
            test = np.concatenate([test,row,np.zeros((row.shape[0]*2,row.shape[1],row.shape[2]))],axis=0)
    
    imsave(f"{images}.jpg",test[:,:,0])
