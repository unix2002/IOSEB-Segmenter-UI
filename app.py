####################################################################################################
#                                                                                                  #
# This Gui has been created for the Globalise project,                                             #
# to improve the ease with which document classification can be performed.                         #
# It is based largely on the Gradio library, and is adaptive to the device it is run on.           #
# For more information, please see the README.md file in the repository.                           #
#                                                                                                  #
# This program has been developed as a part of my thesis project at the University of Amsterdam.   #
#                                                                                                  #
# Author: Joseph Krol                                                                              #
# Date: 2025-05-22                                                                                 #
# License: Apache License 2.0                                                                      #
#                                                                                                  #
####################################################################################################

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import os
import csv
import tempfile

from human_assisted_ui import (
    create_human_assisted_view,
    set_globals_for_human_assisted_mode,
)

MODEL_PATH = "model.pth"
NUM_CLASSES = 5

# Initialize the model
model = mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

# Determine the device to use
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Load the trained model weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)  # Move the model to the chosen device
    model.eval()  # Set the model to evaluation mode (important for inference)
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please check the path.")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 341

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["inside", "outside", "start", "end", "both"]


def classify_single_image_pil(pil_image: Image.Image):
    """
    Takes a PIL Image, preprocesses it, passes it through the model,
    and returns the predicted class name.
    """
    if pil_image is None:
        return "Please upload or provide an image."
    try:
        img_rgb = pil_image.convert("RGB")
        img_tensor = transform(img_rgb).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        predicted_class_name = class_names[predicted_idx.item()]
        return predicted_class_name.capitalize()
    except Exception as e:
        print(f"Error during single image classification: {e}")
        return f"Error: {e}"


set_globals_for_human_assisted_mode(classify_single_image_pil, class_names)


def process_directory_input(directory_data_list_or_path_str):
    """
    Processes a directory input (list of file paths or a single directory path string)
    and returns a CSV file path for download, or None if an error occurs.
    """

    csv_file_path_to_return = None

    image_files_to_process = []
    VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    input_source_name = "the upload"

    if isinstance(directory_data_list_or_path_str, list):
        input_source_name = "the uploaded batch of files"
        for file_path_str in directory_data_list_or_path_str:
            if isinstance(file_path_str, str) and file_path_str.lower().endswith(
                VALID_IMAGE_EXTENSIONS
            ):
                if os.path.isfile(file_path_str):
                    image_files_to_process.append(
                        {
                            "full_path": file_path_str,
                            "relative_path": os.path.basename(file_path_str),
                        }
                    )
                else:
                    print(
                        f"  Warning: Path from list '{file_path_str}' is not a valid file. Skipping."
                    )
            else:
                print(
                    f"  Warning: Item from list '{file_path_str}' is not a string or not a valid image extension. Skipping."
                )

    elif isinstance(directory_data_list_or_path_str, str):
        folder_path_str = directory_data_list_or_path_str
        input_source_name = os.path.basename(folder_path_str)
        if not os.path.isdir(folder_path_str):
            print(f"Error: Path '{folder_path_str}' is not a directory.")
            return None
        for root, _, files in os.walk(folder_path_str):
            for filename in files:
                if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, folder_path_str)
                    image_files_to_process.append(
                        {"full_path": full_path, "relative_path": relative_path}
                    )
    else:
        print(
            f"Error: Unexpected input type for directory processing: {type(directory_data_list_or_path_str)}"
        )
        return None

    if not image_files_to_process:
        print(f"No image files found in {input_source_name}.")
        return None

    print(
        f"Found {len(image_files_to_process)} image(s) to process from {input_source_name}."
    )
    results_for_csv = []
    for image_info in image_files_to_process:
        try:
            pil_img = Image.open(image_info["full_path"])
            classification = classify_single_image_pil(pil_img)
            results_for_csv.append([image_info["relative_path"], classification])
        except Exception as e:
            print(f"  Error processing file {image_info['relative_path']}: {e}")
            results_for_csv.append([image_info["relative_path"], f"Error - {e}"])

    if not results_for_csv:
        print("Could not process any images.")
        return None

    csv_file_object = None
    try:
        csv_file_object = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".csv", newline="", encoding="utf-8"
        )
        csv_path_generated = csv_file_object.name
        with csv_file_object as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["Image_Filename", "Predicted_Class"])
            writer.writerows(results_for_csv)

        csv_file_path_to_return = csv_path_generated
        print(f"CSV file generated at: {csv_file_path_to_return}")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        if csv_file_object and os.path.exists(csv_file_object.name):
            try:
                if not csv_file_object.closed:
                    csv_file_object.close()
                os.remove(csv_file_object.name)
            except OSError:
                pass

    return csv_file_path_to_return


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# IOSEB-Segmenter interface")

    mode_switch = gr.Radio(
        choices=["Single Image", "Directory", "Human Assisted"],
        value="Single Image",
        label="Mode:",
    )

    # --- Single Image Mode UI ---
    with gr.Group(visible=True) as single_image_mode_group:
        single_img_upload_btn = gr.UploadButton(
            "Click to Upload Image", file_types=["image"], file_count="single"
        )
        single_img_display = gr.Image(
            label="Uploaded Image", type="pil", interactive=False
        )
        single_img_classification_output = gr.Textbox(
            label="Classification Result", interactive=False, visible=False
        )

    # --- Directory Mode UI ---
    with gr.Group(visible=False) as directory_mode_group:
        with gr.Column(visible=False) as dir_post_upload_ui:
            dir_process_button = gr.Button("Process Directory and Generate CSV")
            dir_csv_output_file = gr.File(
                label="Download Classification CSV", interactive=False, visible=False
            )

        dir_mode_spacer = gr.Markdown("", visible=False)

        dir_input = gr.File(label="Upload Directory of Images", file_count="directory")

    with gr.Group(visible=False) as human_assisted_mode_group:
        create_human_assisted_view()

    def handle_single_image_upload(uploaded_file_data):
        if uploaded_file_data is None:
            return None, gr.update(value="Please upload an image.", visible=True)

        try:
            pil_image_input = Image.open(uploaded_file_data.name)
            classification = classify_single_image_pil(pil_image_input)
            return pil_image_input, gr.update(value=classification, visible=True)
        except Exception as e:
            print(f"Error opening or processing uploaded single image: {e}")
            return None, gr.update(
                value=f"Error: Could not process image. {e}", visible=True
            )

    single_img_upload_btn.upload(
        fn=handle_single_image_upload,
        inputs=[single_img_upload_btn],
        outputs=[single_img_display, single_img_classification_output],
    )

    def handle_directory_mode_upload(raw_dir_input_data):
        show_process_area = False
        if raw_dir_input_data is not None:
            if isinstance(raw_dir_input_data, list) and len(raw_dir_input_data) > 0:
                show_process_area = True
            elif hasattr(raw_dir_input_data, "name") and os.path.isdir(
                raw_dir_input_data.name
            ):
                show_process_area = True

        return (
            gr.update(visible=show_process_area),
            gr.update(visible=False, value=None),
            gr.update(visible=show_process_area),
        )

    dir_input.upload(
        fn=handle_directory_mode_upload,
        inputs=[dir_input],
        outputs=[dir_post_upload_ui, dir_csv_output_file, dir_mode_spacer],
    )
    dir_input.clear(
        fn=lambda: (
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False),
        ),
        inputs=None,
        outputs=[dir_post_upload_ui, dir_csv_output_file, dir_mode_spacer],
    )

    def handle_directory_mode_button_click(raw_dir_input):

        processed_dir_input = None

        if raw_dir_input is not None:
            if isinstance(raw_dir_input, list):
                processed_dir_input = [
                    f.name
                    for f in raw_dir_input
                    if hasattr(f, "name") and f.name is not None
                ]
            elif hasattr(raw_dir_input, "name") and isinstance(raw_dir_input.name, str):
                if os.path.isdir(raw_dir_input.name):
                    processed_dir_input = raw_dir_input.name
                else:
                    print(
                        f"Warning: Received single FileData object that is not a directory in directory mode: {raw_dir_input.name}"
                    )
                    if raw_dir_input.name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
                    ):
                        processed_dir_input = [raw_dir_input.name]
                    else:
                        return gr.update(value=None, visible=False)
            else:
                print(
                    f"Directory mode received unexpected raw_dir_input structure: {type(raw_dir_input)}"
                )
                return gr.update(value=None, visible=False)
        else:
            print("No directory provided for processing.")
            return gr.update(value=None, visible=False)

        csv_path = process_directory_input(processed_dir_input)

        if csv_path:
            return gr.update(value=csv_path, visible=True)
        else:
            return gr.update(value=None, visible=False)

    dir_process_button.click(
        fn=handle_directory_mode_button_click,
        inputs=[dir_input],
        outputs=[dir_csv_output_file],
    )

    def toggle_ui_visibility(current_mode_selected):
        is_single_mode = current_mode_selected == "Single Image"
        is_directory_mode = current_mode_selected == "Directory"
        is_human_assisted_mode = current_mode_selected == "Human Assisted"

        updates = {
            single_image_mode_group: gr.update(visible=is_single_mode),
            directory_mode_group: gr.update(visible=is_directory_mode),
            human_assisted_mode_group: gr.update(visible=is_human_assisted_mode),
            single_img_display: gr.update(value=None),
            single_img_classification_output: gr.update(value="", visible=False),
            dir_post_upload_ui: gr.update(visible=False),
            dir_csv_output_file: gr.update(value=None, visible=False),
            dir_input: gr.update(value=None),
            dir_mode_spacer: gr.update(visible=False),
        }
        return updates

    mode_switch.change(
        fn=toggle_ui_visibility,
        inputs=[mode_switch],
        outputs=[
            single_image_mode_group,
            directory_mode_group,
            human_assisted_mode_group,
            single_img_display,
            single_img_classification_output,
            dir_post_upload_ui,
            dir_csv_output_file,
            dir_input,
            dir_mode_spacer,
        ],
    )


if __name__ == "__main__":
    header_comment = """
####################################################################################################
#                                                                                                  #
# This Gui has been created for the Globalise project,                                             #
# to improve the ease with which document classification can be performed.                         #
# It is based largely on the Gradio library, and is adaptive to the device it is run on.           #
# For more information, please see the README.md file in the repository.                           #
#                                                                                                  #
# This program has been developed as a part of my thesis project at the University of Amsterdam.   #
#                                                                                                  #
# Author: Joseph Krol                                                                              #
# Date: 2025-05-22                                                                                 #
# License: Apache License 2.0                                                                      #
#                                                                                                  #
####################################################################################################
    """
    print(header_comment)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    iface.launch(server_name="0.0.0.0", server_port=7860)
