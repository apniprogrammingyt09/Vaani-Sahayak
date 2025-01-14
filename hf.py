from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow frontend requests from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define a route to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

# Your previous API and mapping logic here
SIGN_IMAGE_PATH = ""  # Folder with images or videos for each sign
sign_map = {
    'a': 'data/test/A/0.jpg', 'b': 'data/test/B/0.jpg', 'c': 'data/test/C/0.jpg', # and so on...
}

class TextRequest(BaseModel):
    text: str

@app.post("/translate/")
def translate_text(request: TextRequest):
    translated_signs = []

    # Convert text to lowercase to match dictionary keys
    text = request.text.lower()

    # Map each character to its corresponding sign image
    for char in text:
        if char in sign_map:
            translated_signs.append(sign_map[char])
        elif char == " ":
            translated_signs.append("space.png")  # Optional: Add a placeholder for spaces

    return {"signs": translated_signs}