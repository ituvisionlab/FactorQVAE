import os

import gdown
import numpy as np
from PIL import Image

os.makedirs("../datasets", exist_ok=True)
os.makedirs("../datasets/MPI3DComplex", exist_ok=True)

url = "https://drive.google.com/uc?export=download&confirm=pbef&id=1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm"
output = "../datasets/real3d_complicated_shapes_ordered.npz"

gdown.download(url, output, quiet=False)

images = np.load(output)["images"]

for i in range(images.shape[0]):
    print(i)
    pic = Image.fromarray(images[i])
    pic.save("../datasets/MPI3DComplex/{}.png".format(i))



