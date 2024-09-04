import os

import gdown
import h5py
import numpy as np
from PIL import Image

os.makedirs("../datasets", exist_ok=True)
os.makedirs("../datasets/Shapes3D", exist_ok=True)

url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
output = "../datasets/3dshapes.h5"

gdown.download(url, output, quiet=False)

dataset = h5py.File("../datasets/3dshapes.h5", "r")
images = dataset["images"]

for index in range(len(images)):
    print(index)
    np_image = images[index]
    image = Image.fromarray(np_image.astype(np.uint8))
    image.save("../datasets/Shapes3D/{}.png".format(index))



