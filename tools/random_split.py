import pathlib
import os
import tkinter as tk
from tkinter import filedialog
import random
import shutil

root = tk.Tk()
root.withdraw()

raw_src_dir_path = filedialog.askdirectory()
mask_src_dir_path = filedialog.askdirectory()
dest_dir_path = filedialog.askdirectory()

extensions = '*.png', '*jpg', '*jpeg', '*bmp'
raw_src_paths = []
mask_src_paths = []
train = []
val = []
test = []

train_split_value = 1600
val_split_value = train_split_value + 200
test_split_value = val_split_value + 200


def save(img_list, save_location):
    raw_save_loc = create_dir("raw", save_location)
    mask_save_loc = create_dir("labeled", save_location)

    for img_file in img_list:
        raw_file_loc = os.path.abspath(os.path.join(raw_save_loc, img_file.name))
        mask_file = get_mask_file(img_file.stem)
        mask_file_loc = os.path.abspath(os.path.join(mask_save_loc, mask_file.name))
        shutil.copy(img_file.absolute(), raw_file_loc)
        print(mask_file_loc)
        shutil.copy(mask_file.absolute(), mask_file_loc)


def create_dir(dir_name, dest_path):
    dir = os.path.join(dest_path, dir_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def get_mask_file(file_name_without_ext):
    return next(x for x in mask_src_paths if file_name_without_ext in x.name)


# get all img files
for ext in extensions:
    raw_src_paths += pathlib.Path(raw_src_dir_path).glob(ext)
    mask_src_paths += pathlib.Path(mask_src_dir_path).glob(ext)

# shuffle raw files (pseudo random with 420 seed)
random.Random(420).shuffle(raw_src_paths)

# split raw files
train = raw_src_paths[:train_split_value]
val = raw_src_paths[train_split_value:val_split_value]
test = raw_src_paths[val_split_value:test_split_value]

train_dir = create_dir("train", dest_dir_path)
val_dir = create_dir("val", dest_dir_path)
test_dir = create_dir("test", dest_dir_path)

save(train, train_dir)
save(val, val_dir)
save(test, test_dir)