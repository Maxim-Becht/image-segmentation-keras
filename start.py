import json, jsonschema
from jsonschema import Draft4Validator, validators
from pathlib import Path
import tkinter as tk
from tkinter import TclError, ttk, filedialog, messagebox

from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.models.model_utils import transfer_weights

textboxes = {}
json_cfg_path = ""
json_cfg = {}
json_schema_path = ""
json_schema = {}


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

    # validate json file
    ttk.Button(frame, text='Validate JSON cfg', command=lambda: validate_json_CALLBACK()).grid(column=1, row=2)

    # Enable TensorBoard checkbox
    match_case = tk.StringVar(value=1)
    match_case_check = ttk.Checkbutton(
        frame,
        text='Enable TensorBoard',
        variable=match_case,
        command=lambda: print(match_case.get()))
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

    ttk.Button(frame, text='Verificate Datasets').grid(column=0, row=0)
    ttk.Button(frame, text='Visualize Datasets').grid(column=0, row=1)
    ttk.Button(frame, text='Start Training').grid(column=0, row=2)
    ttk.Button(frame, text='Start Evaluation').grid(column=0, row=3)
    ttk.Button(frame, text='Start Prediction').grid(column=0, row=4)
    ttk.Button(frame, text='Cancel').grid(column=0, row=5)

    for widget in frame.winfo_children():
        widget.grid(padx=5, pady=5)

    return frame


def create_main_window():
    root = tk.Tk()
    root.title('Replace')
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
    global json_cfg
    json_cfg_path = choose_json_cfg_path()
    json_cfg = load_json_cfg_file(json_cfg_path)

def choose_json_schema_path_CALLBACK():
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

    print("name", json_cfg["cfg_header"]["name"])


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