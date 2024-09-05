import cv2
from gtts import gTTS
import pygame
import io
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to preprocess image for BLIP model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

# Function to generate caption from image
def generate_caption(image):
    image = preprocess_image(image)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to speak out the caption
def speak_out(text):
    tts = gTTS(text=text, lang='en')
    with io.BytesIO() as mp3_fp:
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        pygame.mixer.music.load(mp3_fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass  # Wait until the speech is done

def process_image(image):
    """Generate caption and speak it out for a given image."""
    try:
        caption = generate_caption(image)
        print("Generated Caption:", caption)
        speak_out(caption)
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Select input method:")
    print("1: Use input image file")
    print("2: Capture from webcam")

    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        input_image_path = input("Enter the path to the image file: ")
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Unable to load image from '{input_image_path}'")
        else:
            process_image(image)

    elif choice == '2':
        video_capture = cv2.VideoCapture(0)  # Use default webcam

        if not video_capture.isOpened():
            print("Error: Unable to open video capture.")
        else:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Unable to capture frame.")
                    break

                # Display the frame using matplotlib
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')  # Turn off axis numbers and ticks
                plt.show(block=False)  # Show image and continue execution

                # Process image from webcam
                process_image(frame)

                # Close the plot to prevent multiple windows
                plt.close()

                # Exit on ESC key press
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Release video capture and close windows
            video_capture.release()
            cv2.destroyAllWindows()

    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
