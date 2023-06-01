from enum import Enum
import json, jsonschema
import subprocess
import os
from jsonschema import Draft4Validator, validators
from pathlib import Path
import tkinter as tk
from tkinter import TclError, ttk, filedialog, messagebox

from keras_segmentation.train import train
from keras_segmentation.predict import predict, predict_multiple, evaluate
from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset
from keras_segmentation.data_utils.visualize_dataset import visualize_segmentation_dataset

from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.models.model_utils import transfer_weights

import gui


class Dataset:
    def __init__(self, images_path, segs_path):
        self.images_path = images_path
        self.segs_path = segs_path


json_cfg_path = ""
json_cfg = {}
json_schema_path = ""
json_schema = {}
json_dataset_cfg = {}
enable_tensorboard_overwrite = False

DS_CAT = Enum('DS_CAT', ['TRAIN', 'VAL', 'TEST'])
IMG_TYPE = Enum('IMG_TYPE', ['RAW', 'LABELED'])


def choose_json_cfg_path():
    cfg_path = filedialog.askopenfilename(filetypes=[("Json", '*.json'), ("All files", "*.*")])
    gui.write_textbox("cfg_path", cfg_path)
    return cfg_path


def choose_json_schema_path():
    schema_path = filedialog.askopenfilename(filetypes=[("Json", '*.json'), ("All files", "*.*")])
    gui.write_textbox("schema_path", schema_path)
    return schema_path  


def load_json_cfg_file(cfg_path):
    json_cfg = {}

    # Opening JSON file
    with open(cfg_path) as json_file:
        json_cfg = json.load(json_file)

        if json_cfg.get("dataset") is not None:
            load_json_dataset_cfg_file(json_cfg["dataset"])
    return json_cfg
    

def load_json_schema_file(schema_path):
    json_schema={}

    # Opening JSON schema file
    with open(schema_path) as json_schema_file:
        json_schema = json.load(json_schema_file)
    return json_schema


def load_json_dataset_cfg_file(dataset_cfg_name):
    global json_dataset_cfg
    json_dataset_cfg_path = Path("./ml_creator/cfgs/datasets/{file}.json".format(file=dataset_cfg_name)).absolute()
    # Opening JSON file
    with open(json_dataset_cfg_path) as json_file:
        json_dataset_cfg = json.load(json_file)
    return json_dataset_cfg


def get_dataset_path(dataset_category: DS_CAT, image_type: IMG_TYPE):
    if dataset_category == DS_CAT.TRAIN:
        if image_type == IMG_TYPE.RAW:
            if json_dataset_cfg:
                return json_dataset_cfg["train_images"]
            else:
                return json_cfg["training"]["train_images"]
        elif image_type == IMG_TYPE.LABELED:
            if json_dataset_cfg:
                return json_dataset_cfg["train_annotations"]
            else:
                return json_cfg["training"]["train_annotations"]
    elif dataset_category == DS_CAT.VAL:
        if image_type == IMG_TYPE.RAW:
            if json_dataset_cfg:
                return json_dataset_cfg["val_images"]
            else:
                return json_cfg["training"]["validation"]["val_images"]
        elif image_type == IMG_TYPE.LABELED:
            if json_dataset_cfg:
                return json_dataset_cfg["val_annotations"]
            else:
                return json_cfg["training"]["validation"]["val_annotations"]
    elif dataset_category == DS_CAT.TEST:
        if image_type == IMG_TYPE.RAW:
            if json_dataset_cfg:
                return json_dataset_cfg["test_images"]
            else:
                return json_cfg["training"]["evaluation"]["images_path"]
        elif image_type == IMG_TYPE.LABELED:
            if json_dataset_cfg:
                    return json_dataset_cfg["test_annotations"]
            else:
                return json_cfg["training"]["evaluation"]["segs_path"]



def load_default_json_schema():
    global json_schema
    schema_path = Path.absolute(Path("./ml_creator/schema.json"))
    gui.write_textbox("schema_path", schema_path) 
    json_schema = load_json_schema_file(schema_path)


def create_cfg_filename():
    if not json_cfg:
        messagebox.showerror("ERROR", "json_cfg not set")
        return
    if json_cfg.get("cfg_header") is None:
        messagebox.showerror("ERROR", "cfg_header in json_cfg not set")
        return

    cfg_header = json_cfg["cfg_header"]
    version = cfg_header["version"]
    version_string = "{major}.{minor}.{bug_fix}".format(major=str(version["major"]), minor=str(version["minor"]), bug_fix=str(version["bug_fix"]))
    cfg_name = "{current_dataset}-{model}-{version}.json".format(current_dataset=cfg_header["dataset"]["current"], model=cfg_header["model"], version=version_string)
    return cfg_name


def auto_rename_cfg_file():
    new_cfg_name = create_cfg_filename()
    if new_cfg_name and json_cfg_path != "":
        new_json_cfg_path = Path.joinpath(Path(json_cfg_path).parent, new_cfg_name)
        os.rename(json_cfg_path, new_json_cfg_path)


def open_tensorboard():
    p = subprocess.Popen(["tensorboard", "--logdir", "logs/fit"])
    # p.terminate()


def validate_json(json_cfg, json_schema):
    def extend_with_default(validator_class):
        validate_properties = validator_class.VALIDATORS["properties"]

        def set_defaults(validator, properties, instance, schema):
            for property, subschema in properties.items():
                if "default" in subschema and not isinstance(instance, list):
                    instance.setdefault(property, subschema["default"])

            for error in validate_properties(
                validator, properties, instance, schema,
            ):
                yield error

        return validators.extend(
            validator_class, {"properties" : set_defaults},
        )

    DefaultValidatingValidator = extend_with_default(Draft4Validator)
    DefaultValidatingValidator(json_schema).validate(json_cfg)

    # re-validate, after potentially adding default values, since default values have to be validated too (type faults)
    jsonschema.validate(
        instance=json_cfg,
        schema=json_schema,
    )


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema and not isinstance(instance, list):
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties" : set_defaults},
    )


#----------CALLBACKS----------# 

def choose_json_cfg_path_CALLBACK():
    global json_cfg_path
    global json_cfg
    json_cfg_path = choose_json_cfg_path()
    json_cfg = load_json_cfg_file(json_cfg_path)


def choose_json_schema_path_CALLBACK():
    global json_schema_path
    global json_schema
    json_schema_path = choose_json_schema_path()
    json_schema = load_json_schema_file(json_schema_path)


def validate_json_CALLBACK():
    if not json_cfg:
        messagebox.showerror("ERROR", "json_cfg not set")
        return

    if not json_schema:
        messagebox.showerror("ERROR", "json_schema not set")
        return

    validate_json(json_cfg, json_schema)


def start_training_CALLBACK():
    #cfg_header = json_cfg["cfg_header"]
    # todo: add header name to writer of TensorBoard file naming
    cfg_training = json_cfg["training"]
    cfg_training_val = cfg_training["validation"]

    # UI driven design:
    # enable_tensorboard = bool(enable_tensorboard_overwrite)

    # overwrite global cfg
    read_image_type = 1
    if cfg_training.get("read_image_type") is not None:
        read_image_type = cfg_training["read_image_type"]

    train (
        model = json_cfg["model_name"],
        train_images = get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.RAW),
        train_annotations = get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.LABELED),
        input_height = json_cfg["input_height"],
        input_width = json_cfg["input_width"],
        n_classes = json_cfg["n_classes"],
        verify_dataset = json_cfg["verify_dataset"],
        checkpoints_path = json_cfg["checkpoints_path"],
        epochs = cfg_training["epochs"],
        batch_size = cfg_training["batch_size"],
        validate = cfg_training["validate"],
        val_images = get_dataset_path(DS_CAT.VAL, IMG_TYPE.RAW),
        val_annotations = get_dataset_path(DS_CAT.VAL, IMG_TYPE.LABELED),
        val_batch_size = cfg_training_val["val_batch_size"],
        auto_resume_checkpoint = cfg_training["auto_resume_checkpoint"],
        load_weights = cfg_training["load_weights"],
        steps_per_epoch = cfg_training["steps_per_epoch"],
        val_steps_per_epoch = cfg_training_val["val_steps_per_epoch"],
        gen_use_multiprocessing = cfg_training_val["gen_use_multiprocessing"],
        ignore_zero_class = json_cfg["ignore_zero_class"],
        optimizer_name = cfg_training["optimizer_name"],
        do_augment = cfg_training["do_augment"],
        augmentation_name = cfg_training["augmentation_name"],
        callbacks = cfg_training["callbacks"],
        custom_augmentation = cfg_training["custom_augmentation"],
        other_inputs_paths = cfg_training["other_inputs_paths"],
        preprocessing = cfg_training["preprocessing"],
        read_image_type = read_image_type,      
                                # cv2.IMREAD_COLOR = 1 (rgb),
                                # cv2.IMREAD_GRAYSCALE = 0,
                                # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
        enable_tensorboard=cfg_training["enable_tensorboard"]
    )


def start_dataset_verification_CALLBACK():
    cfg_preparation = json_cfg["preparation"]
    cfg_verification = cfg_preparation["verification"]

    if cfg_verification.get("datasets") is not None:
        datasets = cfg_verification["datasets"]

        for dataset in datasets:
            verify_segmentation_dataset (
                images_path = dataset["images_path"],
                segs_path = dataset["segs_path"],
                n_classes = json_cfg["n_classes"],
                show_all_errors = cfg_verification["show_all_errors"]
            )
    else:
        datasets = []
        datasets.append(Dataset(get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.RAW), get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.LABELED)))
        datasets.append(Dataset(get_dataset_path(DS_CAT.VAL, IMG_TYPE.RAW), get_dataset_path(DS_CAT.VAL, IMG_TYPE.LABELED)))
        datasets.append(Dataset(get_dataset_path(DS_CAT.TEST, IMG_TYPE.RAW), get_dataset_path(DS_CAT.TEST, IMG_TYPE.LABELED)))

        for dataset in datasets:
            verify_segmentation_dataset (
                images_path = dataset.images_path,
                segs_path = dataset.segs_path,
                n_classes = json_cfg["n_classes"],
                show_all_errors = cfg_verification["show_all_errors"]
            )


def start_dataset_visualization_CALLBACK():
    cfg_preparation = json_cfg["preparation"]
    cfg_visualisation = cfg_preparation["visualisation"]

    image_size = None
    if cfg_visualisation.get("image_size") is not None:
        image_size = cfg_visualisation["image_size"]

    if cfg_visualisation.get("datasets") is not None:
        datasets = cfg_visualisation["datasets"]

        for dataset in datasets:
            visualize_segmentation_dataset (
                images_path=dataset["images_path"],
                segs_path=dataset["segs_path"],
                n_classes=json_cfg["n_classes"],
                do_augment = cfg_visualisation["do_augment"],
                augment_name = cfg_visualisation["augment_name"],
                custom_aug = cfg_visualisation["custom_aug"],
                ignore_non_matching = cfg_visualisation["ignore_non_matching"],
                no_show = cfg_visualisation["no_show"],
                image_size = image_size
            )
    else:
        datasets = []
        datasets.append(Dataset(get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.RAW), get_dataset_path(DS_CAT.TRAIN, IMG_TYPE.LABELED)))
        datasets.append(Dataset(get_dataset_path(DS_CAT.VAL, IMG_TYPE.RAW), get_dataset_path(DS_CAT.VAL, IMG_TYPE.LABELED)))
        datasets.append(Dataset(get_dataset_path(DS_CAT.TEST, IMG_TYPE.RAW), get_dataset_path(DS_CAT.TEST, IMG_TYPE.LABELED)))

        for dataset in datasets:
            visualize_segmentation_dataset (
                images_path=dataset.images_path,
                segs_path=dataset.segs_path,
                n_classes=json_cfg["n_classes"],
                do_augment = cfg_visualisation["do_augment"],
                augment_name = cfg_visualisation["augment_name"],
                custom_aug = cfg_visualisation["custom_aug"],
                ignore_non_matching = cfg_visualisation["ignore_non_matching"],
                no_show = cfg_visualisation["no_show"],
                image_size = image_size
            )


def start_evaluation_CALLBACK():
    cfg_testing = json_cfg["testing"]
    cfg_evaluation = cfg_testing["evaluation"]

    checkpoints_path = json_cfg["checkpoints_path"]
    if cfg_evaluation.get("checkpoints_path") is not None:
        checkpoints_path = cfg_evaluation["checkpoints_path"]

    print(
        evaluate (
        checkpoints_path = checkpoints_path,
	    inp_images_dir = get_dataset_path(DS_CAT.TEST, IMG_TYPE.RAW),
		annotations_dir = get_dataset_path(DS_CAT.TEST, IMG_TYPE.LABELED),
		read_image_type = cfg_evaluation["read_image_type"]))


def start_prediction_CALLBACK():
    cfg_testing = json_cfg["testing"]
    cfg_prediction = cfg_testing["prediction"]

    checkpoints_path = json_cfg["checkpoints_path"]
    if cfg_prediction.get("checkpoints_path") is not None:
        checkpoints_path = cfg_prediction["checkpoints_path"]

    input_path = get_dataset_path(DS_CAT.TEST, IMG_TYPE.RAW),
    input_path_extension = input_path.split('.')[-1]
    output_path = cfg_prediction["output_path"]

    prediction_width = cfg_prediction["prediction_width"]
    if prediction_width <= 0:
        prediction_width = None
    prediction_height = cfg_prediction["prediction_height"]
    if prediction_height <= 0:
        prediction_height = None


    overlay_img = cfg_prediction["overlay_img"]
    show_legends = cfg_prediction["show_legends"]
    class_names = cfg_prediction["class_names"]
    colors = cfg_prediction["colors"]
    read_image_type = cfg_prediction["read_image_type"]

    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return predict (
                inp=input_path, 
                out_fname=output_path,
                checkpoints_path=checkpoints_path,
                prediction_width=prediction_width,
                prediction_height=prediction_height,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                colors=colors,
                read_image_type=read_image_type
            )
    else:
        return predict_multiple (
                inp_dir=input_path, 
                out_dir=output_path,
                checkpoints_path=checkpoints_path,
                prediction_width=prediction_width,
                prediction_height=prediction_height,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                colors=colors,
                read_image_type=read_image_type
            )
    

def open_tensorboard_CALLBACK():
    open_tensorboard()


def auto_rename_cfg_file_CALLBACK():
    auto_rename_cfg_file()


if __name__ == "__main__":
    gui.create_main_window()