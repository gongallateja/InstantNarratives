# InstantNarratives

This project provides a tool for generating captions from images and converting those captions into spoken words. Users can choose between processing a static image file or capturing live video from a webcam. The tool leverages the BLIP (Bootstrapped Language Image Pre-training) model for image captioning and uses Google Text-to-Speech (gTTS) for voice output.

Features:

*Image Captioning: Generates descriptive captions for images using the BLIP model.

*Voice Output: Converts generated captions into spoken words using Google Text-to-Speech (gTTS).

*Flexible Input: Supports both input image files and real-time webcam capture.

*Real-Time Processing: Captions and speaks out captions for each frame in real-time when using a webcam.

Prerequisites: Python 3.x

Required Python libraries: opencv-python gtts pygame Pillow matplotlib transformers torch
