# Player Re Identification in Sports Footage

This project implements a solution for detecting and tracking players in a 15-second video clip (`15sec_input_720p.mp4`), ensuring consistent player identification even when players exit and re-enter the frame. The solution leverages the YOLOv8 object detection model for player detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking, with custom logic for re-identification based on spatial and size similarity.

## Prerequisites

### System Requirements
- **Operating System**: Linux Fedora (or compatible Linux distribution) for local setup; Google Colab for cloud execution
- **Python Version**: Python 3.8 or higher
- **Hardware**: CPU (GPU recommended for faster processing, available on Google Colab with GPU runtime)
- **Storage**: At least 2 GB of free disk space for dependencies, model weights, and output video
- **Input Video**: `15sec_input_720p.mp4` (720p resolution, MP4 format)
- **Model Weights**: Pre-trained YOLOv8 model weights (`best.pt`) trained to detect players
- **Internet Access**: Required for downloading dependencies and `sort.py` (especially on Google Colab)

### Dependencies
The following Python packages are required:
- `ultralytics==8.0.20` (for YOLOv8 object detection)
- `opencv-python-headless==4.5.5.64` (for video processing)
- `numpy==1.23.5` (for numerical computations)
- `scipy==1.9.3` (for Euclidean distance calculations)
- `filterpy==1.4.5` (for Kalman filtering in SORT)

Additionally, a modified version of the SORT algorithm (`sort.py`) is downloaded and used for tracking.

## Setup Instructions

### Option 1: Google Colab Setup
This project was developed and tested on Google Colab, which provides a convenient cloud-based environment with GPU support.

1. **Open Google Colab**
   - Navigate to [Google Colab](https://colab.research.google.com/) and create a new notebook.
   - Optionally, enable GPU runtime: Go to `Runtime > Change runtime type > Hardware accelerator > GPU`.

2. **Upload Input Files**
   - Upload the input video (`15sec_input_720p.mp4`) to the Colab environment:
     ```python
     from google.colab import files
     files.upload()
     ```
   - Upload the pre-trained YOLOv8 model weights (`best.pt`) using the same method.

3. **Install Dependencies**
   - Run the following commands in a Colab cell to install required packages:
     ```bash
     !pip install ultralytics==8.0.20 opencv-python-headless==4.5.5.64 scipy==1.9.3 filterpy==1.4.5
     ```

4. **Download and Modify SORT**
   - Download the SORT algorithm and remove unnecessary GUI/Matplotlib dependencies:
     ```bash
     !wget -O sort.py https://raw.githubusercontent.com/abewley/sort/master/sort.py
     !sed -i '/matplotlib/d' sort.py
     !sed -i '/plotting/d' sort.py
     !sed -i '/TkAgg/d' sort.py
     !sed -i '/skimage/d' sort.py
     ```

### Option 2: Local Fedora Setup
If running on a local Fedora machine, follow these steps:

1. **Prepare the Environment**
   - Create a project directory:
     ```bash
     mkdir player-tracking
     cd player-tracking
     ```
   - Ensure Python 3.8+ is installed:
     ```bash
     python3 --version
     ```
     If not, install it using:
     ```bash
     sudo dnf install python3
     ```

2. **Set Up a Virtual Environment (Recommended)**
   - Create and activate a virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   - Install required packages:
     ```bash
     pip install ultralytics==8.0.20 opencv-python-headless==4.5.5.64 scipy==1.9.3 filterpy==1.4.5
     ```

4. **Download SORT**
   - Download and modify the SORT script:
     ```bash
     wget -O sort.py https://raw.githubusercontent.com/abewley/sort/master/sort.py
     sed -i '/matplotlib/d' sort.py
     sed -i '/plotting/d' sort.py
     sed -i '/TkAgg/d' sort.py
     sed -i '/skimage/d' sort.py
     ```

5. **Prepare Input Files**
   - Place `15sec_input_720p.mp4` and `best.pt` in the project directory.

## Running the Code

### On Google Colab
1. **Copy the Code**
   - Copy the provided Python code (e.g., `player_tracking.py`) into a Colab cell. Ensure the file paths are set correctly:
     ```python
     MODEL_PATH = '/content/best.pt'
     VIDEO_PATH = '/content/15sec_input_720p.mp4'
     OUTPUT_PATH = '/content/output_tracked.mp4'
     ```

2. **Execute the Cell**
   - Run the cell containing the code. The script will:
     - Load the YOLOv8 model and video.
     - Process each frame for player detection and tracking.
     - Save the output video with bounding boxes and player IDs.

3. **Download the Output**
   - After execution, download the output video (`output_tracked.mp4`):
     ```python
     from google.colab import files
     files.download('/content/output_tracked.mp4')
     ```

### On Fedora
1. **Save the Code**
   - Save the Python code as `player_tracking.py` in the project directory.
   - Update file paths if necessary:
     ```python
     MODEL_PATH = 'best.pt'
     VIDEO_PATH = '15sec_input_720p.mp4'
     OUTPUT_PATH = 'output_tracked.mp4'
     ```

2. **Run the Script**
   - Execute the script:
     ```bash
     python3 player_tracking.py
     ```
   - The output video (`output_tracked.mp4`) will be saved in the project directory.

### Expected Output
- The script generates `output_tracked.mp4`, a video with green bounding boxes around detected players and labels (e.g., `Player 1`, `Player 2`) for consistent identification.
- A console message confirms completion: `✅ Done. Video saved to output_tracked.mp4`.
- The output video can be viewed using any media player (e.g., VLC on Fedora).

## Code Structure

The script (`player_tracking.py`) is modular and organized as follows:
- **Step 1: Install Dependencies** - Installs packages and downloads `sort.py` (Colab-specific).
- **Step 2: Imports** - Imports libraries like `ultralytics`, `opencv-python`, and `sort`.
- **Step 3: File Paths** - Defines paths for model, input, and output.
- **Step 4: Load Model** - Initializes the YOLOv8 model.
- **Step 5: Load Video** - Sets up video capture and writer.
- **Step 6: Tracking Variables** - Configures the SORT tracker and dictionaries for player IDs and inactive tracks.
- **Step 7: Processing Loop** - Detects players, tracks them, and re-identifies players using Euclidean distance and bounding box size.
- **Step 8: Cleanup** - Releases resources and saves the output video.

## Notes
- The code assumes the YOLOv8 model (`best.pt`) is trained to detect players with the class name `player`.
- The SORT algorithm uses a Kalman filter and Hungarian algorithm for tracking, with parameters `max_age=120` and `iou_threshold=0.03` (tunable for performance).
- Re-identification matches inactive tracks within 120 frames using Euclidean distance and bounding box size, with a maximum of 25 players.
- The code is optimized for Google Colab but works on Fedora with minor path adjustments.

## Troubleshooting
- **Video Not Found**: Ensure `15sec_input_720p.mp4` is in the project directory and `VIDEO_PATH` is correct.
- **Model Not Found**: Verify `best.pt` is present and `MODEL_PATH` is accurate.
- **Dependency Issues**: Use the specified package versions to avoid compatibility problems.
- **Output Video Not Generated**: Check write permissions for `OUTPUT_PATH` and ensure `opencv-python-headless` is installed.
- **Colab-Specific Issues**: Ensure GPU runtime is enabled for faster processing, and re-upload files if sessions reset.

For a detailed explanation of the approach, methodology, and challenges, refer to the accompanying report (`report.md`).
