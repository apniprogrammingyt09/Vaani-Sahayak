import cv2
import numpy as np
from keras.models import model_from_json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading

# Load the model for Indian Sign Language detection
json_file = open("isl48x481toz.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("isl48x481toz.h5")

# Load GPT-2 model and tokenizer for text suggestion
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")


# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for grayscale input
    return feature / 255.0


# Function to apply skin mask
def apply_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(image, image, mask=mask)


# Function to generate a text suggestion based on the current sentence using GPT-2
def generate_suggestion(current_sentence):
    global next_word
    # Tokenize input sentence
    input_ids = gpt2_tokenizer.encode(current_sentence, return_tensors="pt")

    # Generate continuation using GPT-2
    output = gpt2_model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 5,  # Predict up to 5 additional tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Prevent repetition
        top_k=50,  # Sample from top 50 options
        top_p=0.95,  # Use nucleus sampling
        temperature=0.7,  # Adjust creativity
    )

    # Decode the generated tokens to text
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the predicted next word(s)
    next_word = generated_text[len(current_sentence):].strip().split()[0]  # Take the first word


# Initialize video capture
cap = cv2.VideoCapture(0)

label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

current_sentence = ""
pending_letter = ""
next_word = ""  # Global variable to hold the suggestion

# Main loop for capturing video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply skin mask
    alpha = 1.5  # Contrast control
    beta = 30
    skin_masked_frame = apply_skin_mask(frame)

    # Apply Canny edge detection
    edges = cv2.Canny(skin_masked_frame, 100, 200)

    # Apply ORB feature detection
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(skin_masked_frame, None)
    orb_image = cv2.drawKeypoints(skin_masked_frame, keypoints, None, color=(0, 255, 0))

    # Define a fixed or dynamic region of interest (ROI) rectangle for hand area
    roi_start_x, roi_start_y = 150, 100  # Adjusted for better focus on the hand area
    roi_end_x, roi_end_y = 450, 400
    cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (0, 165, 255), 2)  # Orange frame

    # Extract ROI from the ORB image
    cropframe_orb = orb_image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    cropframe_orb = cv2.resize(cropframe_orb, (48, 48))

    # Convert ORB image to grayscale if needed
    if len(cropframe_orb.shape) == 3:
        cropframe_orb = cv2.cvtColor(cropframe_orb, cv2.COLOR_BGR2GRAY)

    # Extract features and make a prediction using ORB frame
    cropframe_orb = extract_features(cropframe_orb)
    pred = model.predict(cropframe_orb)
    predicted_letter = label[pred.argmax()]

    # Apply a higher confidence threshold to ensure accurate predictions
    if np.max(pred) > 0.6:  # Set to 60% confidence threshold
        pending_letter = predicted_letter
    else:
        pending_letter = 'blank'

    # Display the pending letter without updating the sentence
    accu = "{:.2f}".format(np.max(pred) * 100)
    cv2.putText(frame, f'Pending Letter: {pending_letter}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Sentence: {current_sentence}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display the word suggestion (updated in the background)
    cv2.putText(frame, f'Suggested Word: {next_word}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display images
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Skin Masked Frame", skin_masked_frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Confirm the pending letter by pressing the spacebar
    if key == ord(' '):  # Spacebar pressed to confirm the letter
        if pending_letter == 'blank':
            current_sentence += " "  # Add a space if the predicted letter is 'blank'
        else:
            current_sentence += pending_letter  # Add the letter to the sentence

        # Start a new thread to generate the suggestion asynchronously
        threading.Thread(target=generate_suggestion, args=(current_sentence,)).start()

    # Confirm the suggested word by pressing Enter key (ASCII 13)
    if key == 13:  # Enter key pressed to confirm the suggested word
        current_sentence += " " + next_word  # Add the suggested word to the sentence

    # Exit on 'q' key press
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()