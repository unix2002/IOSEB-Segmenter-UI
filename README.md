# IOSEB-Segmenter UI

This user interface has been created for the Globalise project to improve the ease with which document segmentation can be performed. It is built using the Gradio library and is designed to be adaptive to the device it is run on.

This program was developed as a part of a thesis project at the University of Amsterdam by Joseph Krol (Date: 2025-05-22).

## Features

The UI provides multiple modes for document classification:

* **Single Image Mode:** Upload or drag & drop a single image for immediate classification. The uploaded image and its classification result are displayed.
* **Directory Mode:** Upload a folder/directory of images for batch classification. The results are provided as a downloadable CSV file.
* **Human Assisted Mode:** Upload a directory of images to review the model's initial predictions. Users can navigate through images, confirm correct predictions, or provide corrected classifications. The reviewed results can be downloaded as a CSV file.

## Running the Application

1.  **Prerequisites:**
    * Python 3.11 (or compatible version).
    * `pip` for installing packages.

2.  **Install Dependencies:**
    Navigate to the project directory in your terminal and install the required packages:
    ```bash
    pip install -r requirements.txt

3.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will start on `http://localhost:7860`.

## Project File Structure

A minimal setup includes:

* `app.py`: Main Gradio application script.
* `human_assisted_ui.py`: Defines the UI and logic for the "Human Assisted" mode.
* `model.pth`: The pre-trained PyTorch model for classification.
* `requirements.txt`: Lists Python dependencies.
* `Dockerfile`: Instructions for building the Docker image.
* `README.md`: This file.

## License

This project has been licensed under the apche-2.0 license. See the LICENSE file for more details.
