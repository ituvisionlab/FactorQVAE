import os

import gdown
import zipfile
import shutil

os.makedirs("../datasets", exist_ok=True)
os.makedirs("../datasets/Isaac3D", exist_ok=True)

url = "https://drive.google.com/uc?export=download&confirm=pbef&id=1OmQ1G2wnm6eTsSFGTKFZZAh5D3nQTW1B"
output = "../datasets/isaac3d.zip"

gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("../datasets/Isaac3D")

images_list = os.listdir("../datasets/Isaac3D/Isaac3D_down128/images")

for image in images_list:
    image_name = (image.split("/")[-1]).split(".")[0]
    image_id = int(image_name)
    os.rename("{}/{}".format("../datasets/Isaac3D/Isaac3D_down128/images", image), "{}/{}.png".format("../datasets/Isaac3D", image_id))

shutil.rmtree("../datasets/Isaac3D/Isaac3D_down128")


