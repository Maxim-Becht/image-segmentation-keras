import argparse
import numpy as np
import cv2 as cv
import pathlib
import os

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()


parser = argparse.ArgumentParser(description="Rescale images of folder using nearest neighbor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src", help="Source location")
parser.add_argument("-iw", "--width", help="New image width (absolute value; DefaultValue = 0)")
parser.add_argument("-ih", "--height", help="New image height (absolute value; DefaultValue = 0)")
parser.add_argument("-s", "--scale", help="Scale (Multiplier, for width and height; DefaultValue = 0.5)")
parser.add_argument("-d", "--dest", help="Destination location (only needed if dest != src location")
parser.add_argument("-o", "--override", action="store_true", help="Override existing images")
parser.add_argument("-r", "--recursive", action="store_true", help="Include all images of subdirectories recursively")

args = parser.parse_args()
config = vars(args)


extensions = '*.png', '*jpg', '*jpeg', '*bmp'
recursive = ''
new_width = 0
new_height = 0
scaleFactor = 0.5


if config["src"] == None:
    src_dir_path = filedialog.askdirectory()
else:
    src_dir_path = os.path.abspath(config["src"])

if config["dest"] == None:
    dest_dir_path = os.path.abspath(src_dir_path)
else:
    dest_dir_path = os.path.abspath(config["dest"])
    if not os.path.exists(dest_dir_path):
        os.mkdir(dest_dir_path)

overrideFlag = config["override"]

if config["recursive"]:
    recursive = '**/'

if config["width"] != None:
    new_width = int(config["width"])

if config["height"] != None:
    new_height = int(config["height"])

if config["scale"] != None:
    scaleFactor = float(config["scale"])

print("Rescaling with cfg: ", config)
print("src_dir_path: ", src_dir_path)
print("dest_dir_path: ", dest_dir_path)


for ext in extensions:
    for image_file in pathlib.Path(src_dir_path).glob(recursive + ext):
        img = cv.imread(str(image_file.absolute()))
        
        height, width = img.shape[:2]
        height = int(height * scaleFactor)
        width = int(width * scaleFactor)

        if new_width != 0:
            width = new_width

        if new_height != 0:
            height = new_height
  
        img_res = cv.resize(img, (width, height), interpolation = cv.INTER_NEAREST)


        image_res_file = os.path.join(dest_dir_path, image_file.name)
        if overrideFlag == True:
            cv.imwrite(image_res_file, img_res)
        elif os.path.isfile(image_res_file) == False:
            cv.imwrite(image_res_file, img_res)
