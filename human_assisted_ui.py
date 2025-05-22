import gradio as gr
from PIL import Image
import os
import csv
import tempfile

MODEL_CLASSIFIER_FUNC = None
CLASS_NAMES_LIST = ["inside", "outside", "start", "end", "both"]


def set_globals_for_human_assisted_mode(model_classifier_func, class_names):
    """Sets the model classification function and class names for this module."""
    global MODEL_CLASSIFIER_FUNC, CLASS_NAMES_LIST
    MODEL_CLASSIFIER_FUNC = model_classifier_func
    CLASS_NAMES_LIST = class_names


def _generate_csv_and_get_path(final_reviews_list):
    """Internal helper to generate CSV and return its path or None."""
    if not final_reviews_list:
        print("No reviews to generate CSV from.")
        return None
    try:
        temp_csv_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".csv", newline="", encoding="utf-8"
        )
        csv_path = temp_csv_file.name

        if final_reviews_list:
            fieldnames = final_reviews_list[0].keys()
            writer = csv.DictWriter(temp_csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_reviews_list)

        temp_csv_file.close()
        print(f"Human Assisted CSV generated at: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error generating Human Assisted CSV: {e}")
        if "temp_csv_file" in locals() and os.path.exists(temp_csv_file.name):
            try:
                os.remove(temp_csv_file.name)
            except OSError:
                pass
        return None


def create_human_assisted_view():
    """
    Creates the UI components and logic for the Human Assisted mode.
    """
    with gr.Column() as human_assisted_interface_group:

        image_paths_state = gr.State([])
        current_index_state = gr.State(0)
        reviews_state = gr.State([])

        with gr.Column(visible=True) as upload_area:
            with gr.Row():
                ha_upload_dir = gr.File(
                    label="Upload Directory for Review", file_count="directory", scale=2
                )

        with gr.Column(visible=False) as review_area:
            with gr.Row():
                close_review_button = gr.Button("Close Review")
            with gr.Row(equal_height=False):
                ha_prev_button = gr.Button("⬅️ Previous")
                ha_image_index_info = gr.Markdown(
                    "Image 0 of 0\nfilename.ext", elem_id="ha_image_index_info_elem"
                )
                ha_next_button = gr.Button("Next ")

            ha_current_image_display = gr.Image(
                label="", type="pil", interactive=False, height=400
            )
            ha_model_prediction = gr.Textbox(
                label="Model's Initial Prediction", interactive=False
            )

            with gr.Row() as review_buttons_row:
                ha_review_correct_button = gr.Button("✅ Prediction is Correct")
                ha_review_incorrect_button = gr.Button("❌ Prediction is Incorrect")

            with gr.Column(visible=False) as correction_ui:
                ha_corrected_class_dropdown = gr.Dropdown(
                    choices=CLASS_NAMES_LIST, label="Select Correct Class"
                )
                ha_save_correction_button = gr.Button("Save Corrected Review")

        ha_download_csv_button = gr.DownloadButton(
            "Download Reviewed CSV", visible=False
        )

        def display_image_for_review(paths, index):
            no_image_info_md = "<div style='text-align: center;'>No images to review or index out of bounds.</div>"
            if not paths or not (0 <= index < len(paths)):
                return None, "N/A", no_image_info_md, gr.update(visible=False)

            current_image_path = paths[index]
            filename = os.path.basename(current_image_path)
            try:
                img = Image.open(current_image_path)
                prediction = "Error: Model classifier not set."
                if MODEL_CLASSIFIER_FUNC:
                    prediction = MODEL_CLASSIFIER_FUNC(img)

                index_info_md = f"<div style='text-align: center;'><b>{index + 1} / {len(paths)}</b><br>{filename}</div>"
                return img, prediction, index_info_md, gr.update(visible=False)
            except Exception as e:
                print(f"Error displaying image {current_image_path}: {e}")
                error_info_md = f"<div style='text-align: center;'>Error loading image {index + 1}<br>{filename}</div>"
                return (
                    None,
                    f"Error loading image: {e}",
                    error_info_md,
                    gr.update(visible=False),
                )

        def on_directory_upload(uploaded_dir_data):
            paths = []
            initial_reviews = []
            current_idx = 0

            default_info_md = "<div style='text-align: center;'>Image 0 of 0</div>"
            default_failure_return = (
                [],
                0,
                None,
                "N/A",
                default_info_md,
                [],  # reviews_state
                gr.update(visible=True),  # upload_area
                gr.update(visible=False),  # review_area
                gr.update(visible=False),  # correction_ui
                gr.update(visible=False, value=None),  # download_button
                gr.update(visible=True),  # review_buttons_row
            )

            if uploaded_dir_data is None:
                return default_failure_return

            VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
            if isinstance(uploaded_dir_data, list):
                paths = [
                    f.name
                    for f in uploaded_dir_data
                    if hasattr(f, "name")
                    and f.name is not None
                    and f.name.lower().endswith(VALID_IMAGE_EXTENSIONS)
                ]
            elif hasattr(uploaded_dir_data, "name") and os.path.isdir(
                uploaded_dir_data.name
            ):
                dir_path = uploaded_dir_data.name
                for root, _, files in os.walk(dir_path):
                    for filename in files:
                        if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            paths.append(os.path.join(root, filename))
            paths.sort()

            if not paths:
                return default_failure_return

            img, pred, info, _ = display_image_for_review(paths, current_idx)

            return (
                paths,
                current_idx,  # states
                img,
                pred,
                info,
                initial_reviews,  # ui elements and reviews_state
                gr.update(visible=False),  # upload_area
                gr.update(visible=True),  # review_area
                gr.update(visible=False),  # correction_ui
                gr.update(visible=False, value=None),  # download_button
                gr.update(visible=True),  # review_buttons_row
            )

        ha_upload_dir.upload(
            on_directory_upload,
            inputs=[ha_upload_dir],
            outputs=[
                image_paths_state,
                current_index_state,
                ha_current_image_display,
                ha_model_prediction,
                ha_image_index_info,
                reviews_state,
                upload_area,
                review_area,
                correction_ui,
                ha_download_csv_button,
                review_buttons_row,
            ],
        )

        def close_review_session():
            default_info_md = "<div style='text-align: center;'>Image 0 of 0</div>"
            return (
                [],
                0,
                [],
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "",
                default_info_md,
                gr.update(visible=False),
                gr.update(visible=False, value=None),
                gr.update(value=None),
                gr.update(visible=True),
            )

        close_review_button.click(
            close_review_session,
            inputs=[],
            outputs=[
                image_paths_state,
                current_index_state,
                reviews_state,
                upload_area,
                review_area,
                ha_current_image_display,
                ha_model_prediction,
                ha_image_index_info,
                correction_ui,
                ha_download_csv_button,
                ha_upload_dir,
                review_buttons_row,
            ],
        )

        def navigate_image(paths, current_idx, direction):
            new_idx = current_idx + direction
            if not paths or not (0 <= new_idx < len(paths)):
                new_idx = max(0, min(len(paths) - 1, new_idx)) if paths else 0

            img, pred, info, _ = display_image_for_review(paths, new_idx)
            return (
                new_idx,
                img,
                pred,
                info,
                gr.update(visible=False),
                gr.update(visible=True),
            )

        ha_next_button.click(
            lambda paths, idx: navigate_image(paths, idx, 1),
            inputs=[image_paths_state, current_index_state],
            outputs=[
                current_index_state,
                ha_current_image_display,
                ha_model_prediction,
                ha_image_index_info,
                correction_ui,
                review_buttons_row,
            ],
        )
        ha_prev_button.click(
            lambda paths, idx: navigate_image(paths, idx, -1),
            inputs=[image_paths_state, current_index_state],
            outputs=[
                current_index_state,
                ha_current_image_display,
                ha_model_prediction,
                ha_image_index_info,
                correction_ui,
                review_buttons_row,
            ],
        )

        def process_review_and_advance(
            paths,
            index,
            current_reviews,
            model_pred,
            is_correct_review,
            corrected_class_val=None,
        ):
            current_image_path = paths[index]
            filename = os.path.basename(current_image_path)

            review_entry = {
                "filename": filename,
                "original_prediction": model_pred,
                "reviewed_class": (
                    corrected_class_val if not is_correct_review else model_pred
                ),
                "status": "Correct" if is_correct_review else "Corrected",
            }
            updated_reviews = current_reviews + [review_entry]

            remaining_paths = [p for i, p in enumerate(paths) if i != index]
            new_current_index = index
            if index >= len(remaining_paths):
                new_current_index = max(0, len(remaining_paths) - 1)

            if not remaining_paths:
                generated_csv_path = _generate_csv_and_get_path(updated_reviews)
                all_reviewed_info_md = "<div style='text-align: center;'><b>All images reviewed.</b><br>CSV ready for download if generated.</div>"
                return (
                    [],
                    0,
                    updated_reviews,
                    None,
                    "",
                    all_reviewed_info_md,
                    gr.update(visible=False),
                    gr.update(
                        value=generated_csv_path,
                        visible=True if generated_csv_path else False,
                    ),
                    gr.update(visible=False),
                )
            else:
                img, pred, info, _ = display_image_for_review(
                    remaining_paths, new_current_index
                )
                return (
                    remaining_paths,
                    new_current_index,
                    updated_reviews,
                    img,
                    pred,
                    info,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )

        common_review_outputs = [
            image_paths_state,
            current_index_state,
            reviews_state,
            ha_current_image_display,
            ha_model_prediction,
            ha_image_index_info,
            correction_ui,
            ha_download_csv_button,
            review_buttons_row,
        ]

        ha_review_correct_button.click(
            lambda paths, idx, rev, pred: process_review_and_advance(
                paths, idx, rev, pred, True
            ),
            inputs=[
                image_paths_state,
                current_index_state,
                reviews_state,
                ha_model_prediction,
            ],
            outputs=common_review_outputs,
        )

        def show_correction_ui_fn():
            return gr.update(visible=True)

        ha_review_incorrect_button.click(show_correction_ui_fn, [], [correction_ui])

        ha_save_correction_button.click(
            lambda paths, idx, rev, pred, corrected: process_review_and_advance(
                paths, idx, rev, pred, False, corrected
            ),
            inputs=[
                image_paths_state,
                current_index_state,
                reviews_state,
                ha_model_prediction,
                ha_corrected_class_dropdown,
            ],
            outputs=common_review_outputs,
        )

    return human_assisted_interface_group


if __name__ == "__main__":

    def dummy_classify(pil_img):
        import random

        return random.choice(CLASS_NAMES_LIST).capitalize()

    set_globals_for_human_assisted_mode(dummy_classify, CLASS_NAMES_LIST)

    with gr.Blocks() as demo:
        create_human_assisted_view()
    demo.launch()
