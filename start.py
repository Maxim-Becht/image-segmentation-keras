import json, jsonschema
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

textboxes = {}
json_cfg_path = ""
json_cfg = {}
json_schema_path = ""
json_schema = {}
enable_tensorboard_overwrite = False


def write_textbox(textbox, text):
    textbox.configure(state='normal')
    textbox.insert(1.0,text)
    textbox.configure(state='disabled')

def create_input_frame(container):

    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(0, weight=3)
    frame.columnconfigure(0, weight=1)

    # JSON cfg path text
    ttk.Label(frame, text='JSON cfg path:').grid(column=0, row=0, sticky=tk.W)
    cfg_path_text = tk.Text(frame, width=30, height=1, state="disabled", wrap='none')
    cfg_path_text.grid(column=1, row=0, sticky=tk.W)
    textboxes['cfg_path'] = cfg_path_text

    # JSON schema path text
    ttk.Label(frame, text='JSON schema path:').grid(column=0, row=1, sticky=tk.W)
    schema_path_text = tk.Text(frame, width=30, height=1, state="disabled", wrap='none')
    schema_path_text.grid(column=1, row=1, sticky=tk.W)
    textboxes['schema_path'] = schema_path_text

    # set path buttons
    ttk.Button(frame, text='set', command=lambda: choose_json_cfg_path_CALLBACK()).grid(column=2, row=0)
    ttk.Button(frame, text='set', command=lambda: choose_json_schema_path_CALLBACK()).grid(column=2, row=1)

    # validate json cfg file
    ttk.Button(frame, text='Validate JSON cfg', command=lambda: validate_json_CALLBACK()).grid(column=1, row=2)

    # rename json cfg file
    ttk.Button(frame, text='Auto Rename JSON cfg', command=lambda: auto_rename_cfg_file_CALLBACK()).grid(column=1, row=3)

    # Enable TensorBoard checkbox
    global enable_tensorboard_overwrite 
    enable_tensorboard_overwrite = tk.StringVar(value=1)
    match_case_check = ttk.Checkbutton(
        frame,
        text='Enable TensorBoard',
        variable=enable_tensorboard_overwrite,
        command=lambda: print(enable_tensorboard_overwrite.get()))
    match_case_check.grid(column=0, row=2, sticky=tk.W)

    # Enable Validation checkbox
    wrap_around = tk.StringVar(value=1)
    wrap_around_check = ttk.Checkbutton(
        frame,
        variable=wrap_around,
        text='Validate after epoch',
        command=lambda: print(wrap_around.get()))
    wrap_around_check.grid(column=0, row=3, sticky=tk.W)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)

    return frame


def create_button_frame(container):
    frame = ttk.Frame(container)

    frame.columnconfigure(0, weight=1)

    ttk.Button(frame, text='Verificate Datasets', command=start_dataset_verification_CALLBACK).grid(column=0, row=0)
    ttk.Button(frame, text='Visualize Datasets', command=start_dataset_visualization_CALLBACK).grid(column=0, row=1)
    ttk.Button(frame, text='Start Training', command=start_training_CALLBACK).grid(column=0, row=2)
    ttk.Button(frame, text='Start Evaluation', command=start_evaluation_CALLBACK).grid(column=0, row=3)
    ttk.Button(frame, text='Start Prediction', command=start_prediction_CALLBACK).grid(column=0, row=4)
    ttk.Button(frame, text='Cancel').grid(column=0, row=5)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)

    return frame


def create_main_window():
    root = tk.Tk()
    root.title('UI Controller')
    root.resizable(0, 0)
    try:
        # windows only (remove the minimize/maximize button)
        root.attributes('-toolwindow', True)
    except TclError:
        print('Not supported on your platform')

    # layout on the root window
    root.columnconfigure(0, weight=4)
    root.columnconfigure(1, weight=1)

    input_frame = create_input_frame(root)
    input_frame.grid(column=0, row=0)

    button_frame = create_button_frame(root)
    button_frame.grid(column=1, row=0)

    root.mainloop()


def choose_json_cfg_path():
    cfg_path = filedialog.askopenfilename(filetypes=[("Json", '*.json'), ("All files", "*.*")])
    write_textbox(textboxes['cfg_path'], cfg_path)
    return cfg_path

def choose_json_schema_path():
    schema_path = filedialog.askopenfilename(filetypes=[("Json", '*.json'), ("All files", "*.*")])
    write_textbox(textboxes['schema_path'], schema_path)
    return schema_path

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

def auto_rename_cfg_file_CALLBACK():
    auto_rename_cfg_file()
    

def load_json_cfg_file(cfg_path):
    json_cfg = {}

    # Opening JSON file
    with open(cfg_path) as json_file:
        json_cfg = json.load(json_file)
    return json_cfg
    
def load_json_schema_file(schema_path):
    json_schema={}

    # Opening JSON schema file
    with open(schema_path) as json_schema_file:
        json_schema = json.load(json_schema_file)
    return json_schema


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



def start_training_CALLBACK():
    
    #cfg_header = json_cfg["cfg_header"]
    # todo: add header name to writer of TensorBoard file naming
    cfg_training = json_cfg["training"]
    cfg_training_val = cfg_training["validation"]

    # UI driven design:
    # enable_tensorboard = bool(enable_tensorboard_overwrite)

    # overwrite global cfg
    read_image_type = 1
    if cfg_training["read_image_type"]:
        read_image_type = cfg_training["read_image_type"]


    train (
        model = json_cfg["model_name"],
        train_images = cfg_training["train_images"],
        train_annotations = cfg_training["train_annotations"],
        input_height = json_cfg["input_height"],
        input_width = json_cfg["input_width"],
        n_classes = json_cfg["n_classes"],
        verify_dataset = json_cfg["verify_dataset"],
        checkpoints_path = json_cfg["checkpoints_path"],
        epochs = cfg_training["epochs"],
        batch_size = cfg_training["batch_size"],
        validate = cfg_training["validate"],
        val_images = cfg_training_val["val_images"],
        val_annotations = cfg_training_val["val_annotations"],
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
    datasets =  cfg_verification["datasets"]

    for dataset in datasets:
        verify_segmentation_dataset (
            images_path = dataset["images_path"],
            segs_path = dataset["segs_path"],
            show_all_errors = cfg_verification["show_all_errors"]
        )

def start_dataset_visualization_CALLBACK():

    cfg_preparation = json_cfg["preparation"]
    cfg_visualisation = cfg_preparation["visualisation"]
    datasets =  cfg_visualisation["datasets"]

    for dataset in datasets:
        visualize_segmentation_dataset (
            images_path=dataset["images_path"],
            segs_path=dataset["segs_path"],
            do_augment = cfg_visualisation["do_augment"],
            augment_name = cfg_visualisation["augment_name"],
            custom_aug = cfg_visualisation["custom_aug"],
            ignore_non_matching = cfg_visualisation["ignore_non_matching"],
			no_show = cfg_visualisation["no_show"],
			image_size = cfg_visualisation["image_size"]
        )

def start_evaluation_CALLBACK():

    cfg_testing = json_cfg["testing"]
    cfg_evaluation = cfg_testing["evaluation"]

    checkpoints_path = json_cfg["checkpoints_path"]
    if cfg_evaluation["checkpoints_path"]:
        checkpoints_path = cfg_evaluation["checkpoints_path"]

    evaluate (
        checkpoints_path = checkpoints_path,
	    images_path = cfg_evaluation["images_path"],
		segs_path = cfg_evaluation["segs_path"],
		read_image_type = cfg_evaluation["read_image_type"]
    )

def start_prediction_CALLBACK():

    cfg_testing = json_cfg["testing"]
    cfg_prediction = cfg_testing["prediction"]

    checkpoints_path = json_cfg["checkpoints_path"]
    if cfg_prediction["checkpoints_path"]:
        checkpoints_path = cfg_prediction["checkpoints_path"]

    input_path = cfg_prediction["input_path"]
    input_path_extension = input_path.split('.')[-1]
    output_path = cfg_prediction["output_path"]
    prediction_width = cfg_prediction["prediction_width"]
    prediction_height = cfg_prediction["prediction_height"]
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
    

def create_cfg_filename():
    if not json_cfg:
        messagebox.showerror("ERROR", "json_cfg not set")
        return
    if not json_cfg["cfg_header"]:
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









# replacement = ttk.Entry(frame, width=30)
# replacement.grid(column=1, row=1, sticky=tk.W)


# def main():
#     root = tk.Tk()
#     root.title = ""

#     frame = ttk.Frame(root, padding=(10, 10, 10, 10))
#     frame.grid(column=0, row=0, sticky='swne')

#     # root.withdraw() hide ui
#     root.mainloop()

# path = Path.absolute(Path("./"))
# resolver = validators.RefResolver(
#     base_uri=f"{path.as_uri()}/",
#     referrer=True,
# )
# jsonschema.validate(
#     instance=json_cfg,
#     schema={"$ref": "schema.json"},
#     resolver=resolver,
# )


   # cfg_header = data['cfg_header']

    # model_name = data['model_name']
    # n_classes = data['n_classes']
    # input_height = data['input_height']             #"input_height": "null",
    # input_width = data['input_width']               #"input_width": "null",
    # ignore_zero_class = data['ignore_zero_class']   #"ignore_zero_class": false,
    # verify_dataset = data['verify_dataset']         #"verify_dataset": true,
    # checkpoints_path = data['checkpoints_path']     #"checkpoints_path": "null",
    # read_image_type = data['read_image_type']       #"read_image_type" : 1, 	 

    # training_epochs = data['training']['epochs']

if __name__ == "__main__":
    create_main_window()