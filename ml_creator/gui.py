import tkinter as tk
from tkinter import TclError, ttk, filedialog, messagebox
import api


textboxes = {}


# execute function before going into mainloop
def start():
    api.load_default_json_schema()


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
    ttk.Button(frame, text='set', command=lambda: api.choose_json_cfg_path_CALLBACK()).grid(column=2, row=0)
    ttk.Button(frame, text='set', command=lambda: api.choose_json_schema_path_CALLBACK()).grid(column=2, row=1)

    # validate json cfg file
    ttk.Button(frame, text='Validate JSON cfg', command=lambda: api.validate_json_CALLBACK()).grid(column=1, row=2)

    # rename json cfg file
    ttk.Button(frame, text='Auto Rename JSON cfg', command=lambda: api.auto_rename_cfg_file_CALLBACK()).grid(column=1, row=3)

    # open tensor board
    ttk.Button(frame, text='Open TensorBoard', command=lambda: api.open_tensorboard_CALLBACK()).grid(column=0, row=4)

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

    ttk.Button(frame, text='Verify Datasets', command=lambda: api.start_dataset_verification_CALLBACK()).grid(column=0, row=0)
    ttk.Button(frame, text='Visualize Datasets', command=lambda: api.start_dataset_visualization_CALLBACK()).grid(column=0, row=1)
    ttk.Button(frame, text='Train', command=lambda: api.start_training_CALLBACK()).grid(column=0, row=2)
    ttk.Button(frame, text='Evaluate', command=lambda: api.start_evaluation_CALLBACK()).grid(column=0, row=3)
    ttk.Button(frame, text='Predict', command=lambda: api.start_prediction_CALLBACK()).grid(column=0, row=4)

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

    start()

    root.mainloop()


def write_textbox(textbox_name, text):
    txtbox = textboxes[textbox_name]
    txtbox.configure(state='normal')
    txtbox.insert(1.0,text)
    txtbox.configure(state='disabled')