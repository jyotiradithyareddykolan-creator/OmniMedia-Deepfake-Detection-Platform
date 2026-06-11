import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# CORRECTED IMPORT: Pointing to the services folder
from app.services.detector import run_detection

# Define paths matching our folder structure
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multimodal Deepfake Detection Platform")

# Mount static files and outputs so the HTML can access CSS and images
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Setup Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Renders the main upload page (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    """Handles the file upload, runs the detection, and renders the results."""
    temp_file_path = TEMP_DIR / file.filename
    
    try:
        # 1. Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Pass the file to the detector (which routes to audio or video services)
        result = run_detection(temp_file_path)
        
        # 3. Render the result page with the analysis data
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "result": result}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")
        
    finally:
        # 4. Clean up: Delete the temporary file to save space
        if temp_file_path.exists():
            os.remove(temp_file_path)