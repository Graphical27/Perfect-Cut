from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import os
import uvicorn
from mclass import MultiTaskModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend is running 🚀"}

# Enable CORS so frontend can call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup
state_dict = torch.load("model/main_model.pth", map_location="cpu")
model = MultiTaskModel()
model.load_state_dict(state_dict)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
gender_classes = ["male", "female"]
shape_classes = ["Heart", "Oblong", "Oval", "Round", "Square"]

async def process_image(file: UploadFile):
    # Read uploaded file bytes
    contents = await file.read()
    # Wrap bytes in a file-like object for PIL
    img = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess and run inference
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        g_logits, s_logits = model(x)
        gender = gender_classes[g_logits.argmax(1).item()]
        shape = shape_classes[s_logits.argmax(1).item()]

    return {"gender": gender, "shape": shape}

@app.post("/")
async def root_predict(file: UploadFile = File(...)):
    """
    Root POST endpoint to handle file uploads from the frontend
    """
    return await process_image(file)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Original prediction endpoint, kept for backward compatibility
    """
    return await process_image(file)

# Add this section to run the app with the correct port binding
if __name__ == "__main__":
    # Get port from environment variable or default to 10000 (Render's default)
    port = int(os.environ.get("PORT", 10000))
    # Run the FastAPI app with uvicorn, binding to 0.0.0.0 to accept all incoming connections
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
