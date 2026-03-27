Deepfake Video Detection using FaceForensics++
A deep learning pipeline designed to detect manipulated "deepfake" videos. This project leverages the FaceForensics++ dataset and uses a frame-sampling approach to efficiently classify videos as either pristine (real) or manipulated (fake) without the computational overhead of processing every single frame.

🧠 Approach & Methodology
Processing entire videos frame-by-frame is computationally expensive and often redundant. This model uses a randomized frame-sampling strategy:

Frame Extraction: During both the training and testing phases, the pipeline randomly extracts exactly 8 frames from the target video.

Feature Extraction: These 8 frames are preprocessed (faces isolated, resized, and normalized) and passed through a Convolutional Neural Network (CNN).

Classification: The model evaluates the spatial artifacts in these individual frames. The final prediction for the video is determined by aggregating the results of the 8 frames (e.g., via majority voting or averaging the confidence scores).

📊 Dataset
This project is trained on the FaceForensics++ Dataset, a large-scale dataset of manipulated face videos.

Note: You must request access from the FaceForensics++ creators to download the dataset.

The data is typically split into Real and Fake (manipulated via Deepfakes, Face2Face, FaceSwap, etc.) directories.

🛠️ Tech Stack & Requirements
Language: Python 3.x

Deep Learning Framework: TensorFlow / Keras

Computer Vision: OpenCV (cv2) for frame extraction and face detection.

Data Manipulation: NumPy, Pandas

Install the dependencies:

Bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
🚀 Installation & Setup
1. Clone the repository:

Bash
git clone https://github.com/abhijitsantra01/deepfake-detection.git
cd deepfake-detection
2. Prepare the Data:

Download the FaceForensics++ dataset.

Organize your directory structure as follows:

Plaintext
dataset/
├── train/
│   ├── real/
│   └── fake/
├── test/
│   ├── real/
│   └── fake/
3. Train the Model:
Run the training script. This will automatically extract 8 random frames from each training video on the fly or pre-process them into a designated folder.

Bash
python train.py --data_dir ./dataset/train --epochs 20 --batch_size 16
🎥 Testing / Inference
To test the model on a single video sample, run the inference script. The system will grab 8 random frames from the input video, pass them through the trained model, and output a final Real or Fake prediction.

Bash
python predict.py --video_path ./sample_video.mp4 --model_weights ./saved_models/deepfake_model.h5
📈 Future Improvements
Temporal Analysis: Incorporate an LSTM or GRU layer on top of the CNN to analyze the sequence and temporal inconsistencies between the 8 frames, rather than just treating them as independent images.

Advanced Face Tracking: Implement MTCNN or MediaPipe for more robust face cropping before feeding the frames into the classifier.

Dynamic Frame Selection: Instead of purely random extraction, implement a structural similarity index (SSIM) check to ensure the 8 frames are visually distinct from one another.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
