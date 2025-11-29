import os
import cv2
import tempfile
import json
import uuid
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any

# --- GEMINI IMPORTS ---
from google import genai
from google.genai.errors import APIError
from google.genai.types import Part

# ------------------------
# Config
# ------------------------
# --- GEMINI CLIENT SETUP ---
try:
    # Use api_key from environment, or let it fail cleanly
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Warning: Gemini client initialization failed. Is GEMINI_API_KEY set? Error: {e}")
    client = None

MODEL_NAME = "gemini-2.5-flash"
# Increased for complex HTML output and multiple image inputs
MAX_OUTPUT_TOKENS = 8000 
STATIC_ROOT_DIR = os.path.join(os.getcwd(), "static")
VIDEO_DIR = os.path.join(STATIC_ROOT_DIR, "videos")
SAMPLE_VIDEO_PATH = "/static/videos/sample_video.mp4"

# Ensure directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)


# ------------------------
# App setup
# ------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only mount static directory for videos and HTML assets
app.mount("/static", StaticFiles(directory=STATIC_ROOT_DIR), name="static")

# ------------------------
# Frontend HTML (SYNTAX FIXED: All literal {} changed to {{}})
# ------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Artistic Swimming Analyzer</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            /* GENERAL STYLES AND COLORS */
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background-color: #f4f7f6;
                color: #333;
            }}
            .container {{
                /* Container width remains the same */
                max-width: 1200px;
                margin: 20px auto;
                padding: 20px;
                background: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }}
            h2 {{
                color: #007bff;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 5px;
                margin-top: 25px;
            }}
            h3 {{
                margin-top: 15px;
                color: #555;
            }}
            button {{
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
                margin-left: 10px;
            }}
            button:hover {{
                background-color: #218838;
            }}
            #useSampleBtn {{
                background-color: #ffc107;
                color: #333;
            }}
            #useSampleBtn:hover {{
                background-color: #e0a800;
            }}
            /* Disabled button style */
            button[disabled] {{
                background-color: #cccccc !important; 
                cursor: not-allowed;
            }}

            /* UPDATED: CSS for the image header (60% height) */
            #headerLogo {{
                /* Stretch to full width of the container */
                width: 100%;
                /* Height/size reduction is typically done via max-height or max-width/width/height */
                max-height: 200px; /* Using max-width to effectively reduce the size */
                height: auto;
                object-fit: fill;
                display: block; 
                margin: 0 auto 20px auto; 
            }}


            /* FRAME STYLES (3-COLUMN GRID) */
            #framesContainer {{
                display: grid; 
                /* Explicitly setting three equal-width columns */
                grid-template-columns: 1fr 1fr 1fr; 
                gap: 20px; 
                margin-bottom: 20px;
            }}
            .frame-box {{ 
                display: flex;
                flex-direction: column; 
                justify-content: space-between; 
                align-items: center;
                border: 2px solid #cce5ff;
                border-radius: 8px;
                background-color: #eaf3ff;
                padding: 10px;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            /* POP UP/DOWN EFFECT */
            .frame-box.focused {{
                border-color: #007bff;
                box-shadow: 0 0 15px rgba(0, 123, 255, 0.5);
                transform: scale(1.05); 
                background-color: #ccddff;
                z-index: 10; 
            }}
            img {{ 
                width: 100%; 
                height: auto; 
                margin-bottom: 5px;
                border: 1px solid #007bff;
                border-radius: 4px;
            }}
            .frame-info {{
                font-size: 0.8em; 
                color: #666;
                margin: 5px 0;
            }}
            .focus-btn {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                text-align: center;
                font-size: 0.9em;
                margin-top: 5px;
                transition: background-color 0.2s;
            }}
            .focus-btn.selected {{
                background-color: #dc3545; 
            }}
            
            /* VERTICAL COMPACTION */
            .control-group {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 20px;
            }}


            /* AI JUDGEMENT RESPONSE STYLES */
            #serverResponse {{
                border: 2px solid #ffc107;
                padding: 15px; 
                background: #fffbe6;
                border-radius: 6px;
            }}
            #serverResponse table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 10px 0; 
                font-size: 0.9em;
            }}
            #serverResponse th, #serverResponse td {{ 
                border: 1px solid #ccc; 
                padding: 8px; 
                text-align: left; 
            }}
            #serverResponse th {{ 
                background-color: #007bff;
                color: white; 
            }}
            input[type="file"], textarea, select {{
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-bottom: 10px;
                width: 100%;
                box-sizing: border-box;
            }}
            #figureSelect {{
                width: 250px;
            }}
            #sampleVideoPlayer {{
                width: 100%;
                max-width: 400px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
    <div class="container">
        <img id="headerLogo" src="/static/videos/Judging bot title.png" alt="Artistic Swimming Judging Bot Title"/>

        <h2>Upload Video 🎥</h2>
        
        <video id="sampleVideoPlayer" controls>
            <source src="{SAMPLE_VIDEO_PATH}" type="video/mp4">
            Your browser does not support the video tag.
        </video>

        <div class="control-group">
            <input type="file" id="videoInput" accept="video/*" style="margin-bottom: 0; width: auto;">
            <button id="uploadBtn" disabled>Upload Video and Extract Frames</button>
            <button id="useSampleBtn">Use Sample Video</button>
        </div>

        <hr/>
        <h2>Extracted Key Frames 📸</h2>
        <div id="framesContainer"></div>

        <hr/>
        <h2>Figure Details 📝</h2>
        <h3>Observations (optional)</h3>
        <textarea id="obsBox" rows="4" placeholder="Add specific notes about the athlete or attempt..."></textarea>

        <h3>Select Figure & Judge</h3>
        <div class="control-group">
            <select id="figureSelect">
                <option>301 Barracuda</option>
                <option>309 Reverse Barracuda</option>
                <option>360 Catalina</option>
                <option>402 Ballet Leg</option>
                <option>420 Dolphin</option>
                <option>440 Somersault Back</option>
                <option>502 Crane</option>
                <option>Other</option>
            </select>

            <button id="submitSelected" style="margin-top:0;">Send Selected Frames for Judgement</button>
        </div>

        <hr/>
        <h2>Artistic Swimming Figure Judging Bot 🤖</h2>
        <div id="serverResponse" style="min-height: 100px;">Awaiting Judgement...</div>

        <script>
        // *** FRAME DATA NOW STORES BASE64 STRING ***
        let extractedFrames = []; // [{{ base64_data: '...', timestamp_sec: '...' }}, ...]
        let videoFileToUpload = null;
        let selectedFrameIndices = new Set(); 

        function checkReadyState() {{
            const uploadBtn = document.getElementById("uploadBtn");
            const fileSelected = document.getElementById("videoInput").files.length > 0;
            const sampleReady = videoFileToUpload !== null;

            uploadBtn.disabled = !(fileSelected || sampleReady);
        }}

        function toggleFrameFocus(frameIndex) {{
            const frameBox = document.getElementById(`frame-box-${{frameIndex}}`);
            const focusBtn = document.getElementById(`focus-btn-${{frameIndex}}`);

            if (selectedFrameIndices.has(frameIndex)) {{
                selectedFrameIndices.delete(frameIndex);
                frameBox.classList.remove('focused');
                focusBtn.classList.remove('selected');
                focusBtn.textContent = 'Focus / Select';
            }} else {{
                selectedFrameIndices.add(frameIndex);
                frameBox.classList.add('focused');
                focusBtn.classList.add('selected');
                focusBtn.textContent = 'Selected';
            }}
        }}

        // --- Event Listener for Sample Button ---
        document.getElementById("useSampleBtn").onclick = async () => {{
            document.getElementById("serverResponse").innerHTML = "Loading sample video file...";
            
            try {{
                const response = await fetch('{SAMPLE_VIDEO_PATH}');
                const blob = await response.blob();
                
                videoFileToUpload = new File([blob], "sample_video.mp4", {{type: "video/mp4"}});
                
                document.getElementById("videoInput").value = null;
                document.getElementById("serverResponse").innerHTML = "Sample video ready. Click 'Upload Video and Extract Frames'.";
                
                checkReadyState();

            }} catch (e) {{
                document.getElementById("serverResponse").innerHTML = `Error loading sample video. Ensure the file exists.`;
                videoFileToUpload = null;
                console.error("Error fetching sample video:", e);
                checkReadyState();
            }}
        }};

        // --- Event Listener for Upload Button ---
        document.getElementById("uploadBtn").onclick = async () => {{
            let fileToProcess = videoFileToUpload || document.getElementById("videoInput").files[0];

            if (!fileToProcess) {{ 
                document.getElementById("serverResponse").innerHTML = "Error: No video file is selected or loaded.";
                return; 
            }}
            document.getElementById("serverResponse").innerHTML = "Extracting frames...";

            const formData = new FormData();
            formData.append("video", fileToProcess);

            // API Call: Expects Base64 image data back
            const res = await fetch("/extract_frames", {{ method:"POST", body: formData }});
            const data = await res.json();

            // Store Base64 data and timestamp
            extractedFrames = data.frames || [];
            selectedFrameIndices.clear(); 
            renderFrames();
            document.getElementById("serverResponse").innerHTML = "Frames extracted. Select key frames using the 'Focus' button and press 'Send' to continue.";
        }};

        // --- Handle file input change ---
        document.getElementById("videoInput").onchange = () => {{
            videoFileToUpload = null; 
            checkReadyState();
        }};

        // --- Render Frames Function (MODIFIED to use Base64 as Image Source) ---
        function renderFrames() {{
            const container = document.getElementById("framesContainer");
            container.innerHTML = "";
            
            extractedFrames.forEach((f, idx) => {{
                const div = document.createElement("div");
                div.className = "frame-box";
                div.id = `frame-box-${{idx}}`;
                
                // Use Base64 data URI as the image source
                div.innerHTML = `
                    <img src="data:image/jpeg;base64,${{f.base64_data}}" /> 
                    <div class="frame-info">Time: ${{f.timestamp_sec}}s</div> 
                    <button id="focus-btn-${{idx}}" class="focus-btn" onclick="toggleFrameFocus(${{idx}})">Focus / Select</button>
                `;
                container.appendChild(div);
            }});
        }}

        // --- Submit Judgement Function (MODIFIED to send Base64 data) ---
        document.getElementById("submitSelected").onclick = async () => {{
            const selectedIndices = Array.from(selectedFrameIndices).sort((a, b) => a - b);
            if (selectedIndices.length === 0) {{ alert("Select at least one frame using the Focus button."); return; }}
            document.getElementById("serverResponse").innerHTML = "Sending frames for AI Judgement... Please wait.";


            // Create an array of Base64 strings to send to the backend
            const selectedBase64Data = selectedIndices.map(idx => {{
                return extractedFrames[idx].base64_data;
            }});

            const formData = new FormData();
            formData.append("figure_name", document.getElementById("figureSelect").value);
            formData.append("observations", document.getElementById("obsBox").value);
            // Send the Base64 array as a JSON string
            formData.append("frame_base64_json", JSON.stringify(selectedBase64Data));

            // NOTE: API endpoint changed to /judge_base64_frames
            const res = await fetch("/judge_base64_frames", {{ method:"POST", body: formData }});
            const data = await res.json();
            
            const serverResponseDiv = document.getElementById("serverResponse");
            try {{
                // Note: The LLM output MUST use the HTML <table> format as instructed.
                serverResponseDiv.innerHTML = marked.parse(data.llm_output || "");
            }} catch (e) {{
                serverResponseDiv.textContent = JSON.stringify(data, null, 2);
                console.error("Error parsing LLM output:", e);
            }}
        }};
        
        document.addEventListener('DOMContentLoaded', checkReadyState);
        </script>
    </div>
    </body>
    </html>
    """

# ------------------------
# Extract frames Endpoint (Base64 Output)
# ------------------------
@app.post("/extract_frames")
async def extract_frames(video: UploadFile = File(...)):
    # Use temporary file to process video (necessary with cv2)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"frames": [], "message": "Could not open video file."})

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        MAX_WIDTH = 800
        target_frames = 6
        step = max(1, total_frames // 10)

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if count % step == 0:
                # Simple motion/detail check
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.std() >= 10:

                    # --- RESIZE LOGIC ---
                    height, width = frame.shape[:2]
                    resized_frame = frame
                    if width > MAX_WIDTH:
                        ratio = MAX_WIDTH / width
                        new_width = MAX_WIDTH
                        new_height = int(height * ratio)
                        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    # Encode resized frame to JPEG bytes with compression 75
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
                    _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
                    
                    # Convert bytes to Base64 string
                    base64_encoded = base64.b64encode(buffer).decode('utf-8')

                    frames_data = {
                        "base64_data": base64_encoded,
                        "timestamp_sec": round(count/fps, 2)
                    }

                    # --- SLIDING WINDOW LOGIC (Keeps the latest `target_frames` frames) ---
                    if len(frames) >= target_frames:
                        frames.pop(0) 
                    frames.append(frames_data)
                    # --- END SLIDING WINDOW LOGIC ---

            count += 1
    finally:
        cap.release()
        os.remove(path)
        
    return {"frames": frames}


# ------------------------
# Judge frames with LLM Endpoint (Base64 Input)
# ------------------------
@app.post("/judge_base64_frames")
async def judge_frames(
    figure_name: str = Form(...),
    observations: str = Form(""),
    frame_base64_json: str = Form(...) # Now expects Base64 strings
):
    global client
    if not client:
        return JSONResponse(status_code=500, content={"llm_output": "Error: Gemini client not initialized. Check GEMINI_API_KEY."})
    
    frame_base64_list: List[str] = json.loads(frame_base64_json)

    # Load prompt template from JSON file (retained for structured judging)
    try:
        with open("as_judging.json") as f:
            prompt_template = json.load(f)
    except FileNotFoundError:
        print("WARNING: as_judging.json not found. Using default guidelines.")
        prompt_template = {"content": "Apply standard Artistic Swimming rules for technical execution and scoring."}
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"llm_output": f"Error: Invalid JSON in as_judging.json: {e}"})

    # 1. Prepare text prompt and image data
    gemini_content: List[Any] = []
    
    prompt_text = (
        f"You are an expert Artistic Swimming judge. Analyze the sequence of {len(frame_base64_list)} images for the figure: '{figure_name}'. "
        f"Observations: '{observations}'. "
        "Calculate the score based on the three key transitions (T1, T2, T3) inherent in this figure. "
        f"Reference the following judging guidelines: {prompt_template.get('content', 'No guidelines provided')}. "
        
        # *** STRICT FORMATTING REQUIREMENT (Retained) ***
        "**CRITICAL INSTRUCTION: THE ENTIRE OUTPUT MUST BE FORMATTED EXACTLY LIKE THE EXAMPLE PROVIDED TO YOU. "
        "The assessment MUST start with the score summary, followed by a single HTML <table> with the required six columns: "
        "Transition, Max NVT, Max PV, Awarded PV, Awarded NVT, Key Observations. "
        "Do NOT use Markdown tables. Only use the HTML <table> format.** "
        "End the response with a 'Deductions' list and a 'What to Improve' list with numerical PV points."
    )
    gemini_content.append(prompt_text)

    # 2. Convert Base64 strings back to binary Parts for Gemini
    for b64_data in frame_base64_list:
        try:
            image_bytes = base64.b64decode(b64_data)
            # Create the Gemini Part object directly from binary data
            image_part = Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
            gemini_content.append(image_part)
        except Exception as e:
            # Skip corrupted Base64 or decoding errors
            print(f"Error decoding Base64 image: {e}")
            continue

    files_processed = len(frame_base64_list)
    print(f"Sending {files_processed} Base64 images and complex, strictly-formatted prompt to {MODEL_NAME}...")

    if files_processed == 0:
        return JSONResponse(status_code=400, content={"llm_output": "Error: No frames were processed for the model."})

    # --- GEMINI API Call & Response Handling ---
    output_text = ""
    try:
        completion = client.models.generate_content(
            model=MODEL_NAME, 
            contents=gemini_content,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
        )
        
        output_text = completion.text
        
        if not output_text:
            finish_reason = completion.candidates[0].finish_reason.name if completion.candidates else "UNKNOWN"
            if finish_reason == 'SAFETY':
                print("WARNING: Gemini blocked the response due to safety filters.")
                output_text = "## 🚨 Response Blocked by Safety Filters\n\nTry adjusting your prompt or selecting different frames."
            else:
                print(f"WARNING: Gemini returned a blank response string. Finish Reason: {finish_reason}")
                output_text = f"The AI returned a blank response (Reason: {finish_reason}). Check the server logs for details."
        else:
            print(f"Gemini API call successful. First 100 chars: {output_text[:100]}...") 

    except APIError as e:
        output_text = f"Gemini API call failed (APIError). Status: {e.status_code}. Details: {e.message}"
        print(f"FATAL LLM API ERROR: {e}")
    except Exception as e:
        output_text = f"LLM call failed (General Exception): {type(e).__name__}: {e}"
        print(f"FATAL LLM ERROR: {e}")

    return {
        "llm_output": output_text,
        "num_frames": files_processed,
        "figure_name": figure_name,
        "observations": observations
    }