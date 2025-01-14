from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import Body
from pydantic import BaseModel
import cv2
import numpy as np
from keras.models import model_from_json
from transformers import pipeline
import threading

app = FastAPI()

# Load sign language detection model
with open("isl48x481toz.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("isl48x481toz.h5")

# Load BERT for text suggestion (replace with Hindi model if needed)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Labels for model predictions
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

# Thread-safe variable for next word suggestion
lock = threading.Lock()
next_word = ""
current_sentence = ""

# Function to generate text suggestion asynchronously
def generate_suggestion(current_sentence):
    global next_word
    masked_sentence = current_sentence + " [MASK]"
    suggestion = fill_mask(masked_sentence)[0]['token_str']

    with lock:
        next_word = suggestion

# Function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to apply skin mask
def apply_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(image, image, mask=mask)

# Pydantic model for confirmation requests
class ConfirmRequest(BaseModel):
    pending_letter: str

@app.post("/confirm")
async def confirm_letter(request: ConfirmRequest):
    global current_sentence
    if request.pending_letter != "blank":
        with lock:
            current_sentence += request.pending_letter + ' '
    return {"current_sentence": current_sentence}

# HTML for UI
@app.get("/")
async def get():
    html_content = """
    <html>
    <head>
        <title>Sign Language Detection</title>
        <style>
            #overlay {
                position: absolute;
                top: 0;
                left: 0;
            }
        </style>
    </head>
    <body>
        <h1>Real-Time Sign Language Detection</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="overlay" width="640" height="480"></canvas>
        <p>Detected Sentence: <span id="sentence"></span></p>
        <p>Pending Letter: <span id="pending_letter"></span></p>
        <p>Word Suggestion: <span id="suggestion"></span></p>
        <script>
        const video = document.getElementById('video');
        const overlay = document.getElementById('overlay');
        const context = overlay.getContext('2d');
        const ws = new WebSocket('ws://localhost:8000/ws');
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            function captureAndSendFrame() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob((blob) => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        ws.send(reader.result);
                    };
                    reader.readAsArrayBuffer(blob);
                }, 'image/jpeg');
                drawROI();
                setTimeout(captureAndSendFrame, 100);
            }
            captureAndSendFrame();
        });

        function drawROI() {
            context.clearRect(0, 0, overlay.width, overlay.height);
            context.strokeStyle = 'red';
            context.lineWidth = 2;
            context.strokeRect(150, 100, 300, 300);
        }

        ws.onmessage = function(event) {
            const result = JSON.parse(event.data);
            document.getElementById('sentence').textContent = result.sentence;
            document.getElementById('pending_letter').textContent = result.pending_letter;
            document.getElementById('suggestion').textContent = result.suggestion;
        };

        // Function to confirm predicted class
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space') { // Use space key to confirm
                const pendingLetter = document.getElementById('pending_letter').textContent;
                if (pendingLetter !== "blank") {
                    // Send the pending letter to the server for confirmation
                    fetch('/confirm', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ pending_letter: pendingLetter })
                    });
                }
            }
        });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# WebSocket endpoint for video stream and detection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global current_sentence
    pending_letter = ""

    while True:
        try:
            frame_data = await websocket.receive_bytes()
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            skin_masked_frame = apply_skin_mask(frame)
            roi_start_x, roi_start_y, roi_end_x, roi_end_y = 150, 100, 450, 400
            cropframe = skin_masked_frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
            cropframe = cv2.resize(cropframe, (48, 48))
            cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)

            features = extract_features(cropframe)
            pred = model.predict(features)
            predicted_letter = LABELS[pred.argmax()]

            if np.max(pred) > 0.9:
                pending_letter = predicted_letter
            else:
                pending_letter = "blank"

            # Generate suggestion in background
            threading.Thread(target=generate_suggestion, args=(current_sentence,)).start()

            # Send the detection result back to client
            result = {
                "sentence": current_sentence,
                "pending_letter": pending_letter,
                "suggestion": next_word
            }
            await websocket.send_json(result)

        except Exception as e:
            print(f"WebSocket error: {e}")
            break