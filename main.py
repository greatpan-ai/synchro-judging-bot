import os
import cv2
import tempfile
import json
import uuid
import time
import base64 # Retaining base64 import but not used for Gemini Part objects
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# --- GEMINI IMPORTS ---
from google import genai
from google.genai.errors import APIError
from typing import List

# ------------------------
# Config
# ------------------------
# --- GEMINI CLIENT SETUP ---
# Client will automatically look for the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    # This will be caught if the key is missing during local dev
    print(f"Warning: Gemini client initialization failed. Is GEMINI_API_KEY set? Error: {e}")
    client = None # Set to None for error handling later

MODEL_NAME = "gemini-2.5-pro" # Recommended for high-quality, complex multimodal reasoning
FRAME_DIR = "temp_frames"
STATIC_ROOT_DIR = os.path.join(os.getcwd(), "static")
VIDEO_DIR = os.path.join(STATIC_ROOT_DIR, "videos")
SAMPLE_VIDEO_PATH = "/static/videos/sample_video.mp4"

os.makedirs(FRAME_DIR, exist_ok=True)
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

app.mount("/frames", StaticFiles(directory=FRAME_DIR), name="frames")
app.mount("/static", StaticFiles(directory=STATIC_ROOT_DIR), name="static")


# ------------------------
# Frontend HTML 🎨
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
        let extractedFrames = [];
        let videoFileToUpload = null;
        let selectedFrameIndices = new Set(); 

        // --- Function to check if a file is ready and enable the button ---
        function checkReadyState() {{
            const uploadBtn = document.getElementById("uploadBtn");
            const fileSelected = document.getElementById("videoInput").files.length > 0;
            const sampleReady = videoFileToUpload !== null;

            if (fileSelected || sampleReady) {{
                uploadBtn.disabled = false;
            }} else {{
                uploadBtn.disabled = true;
            }}
        }}

        // --- Frame Selection Logic (Pop Up/Down) ---
        function toggleFrameFocus(frameIndex) {{
            const frameBox = document.getElementById(`frame-box-${{frameIndex}}`);
            const focusBtn = document.getElementById(`focus-btn-${{frameIndex}}`);

            if (selectedFrameIndices.has(frameIndex)) {{
                // Deselect
                selectedFrameIndices.delete(frameIndex);
                frameBox.classList.remove('focused');
                focusBtn.classList.remove('selected');
                focusBtn.textContent = 'Focus / Select';
            }} else {{
                // Select
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
                
                // Set videoFileToUpload variable
                videoFileToUpload = new File([blob], "sample_video.mp4", {{type: "video/mp4"}});
                
                // Clear the file input visually (security prevents setting the value)
                document.getElementById("videoInput").value = null;
                document.getElementById("serverResponse").innerHTML = "Sample video ready. Click 'Upload Video and Extract Frames'.";
                
                // Check and enable the button
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

            const res = await fetch("/extract_frames", {{ method:"POST", body: formData }});
            const data = await res.json();

            extractedFrames = data.frames || [];
            selectedFrameIndices.clear(); 
            renderFrames();
            document.getElementById("serverResponse").innerHTML = "Frames extracted. Select key frames using the 'Focus' button and press 'Send' to continue.";
        }};

        // --- Handle file input change ---
        document.getElementById("videoInput").onchange = () => {{
            // Clear the sample video reference if a new file is chosen
            videoFileToUpload = null; 
            // Check and enable/disable the button
            checkReadyState();
        }};

        // --- Render Frames Function ---
        function renderFrames() {{
            const container = document.getElementById("framesContainer");
            container.innerHTML = "";
            
            extractedFrames.forEach((f, idx) => {{
                const div = document.createElement("div");
                div.className = "frame-box";
                div.id = `frame-box-${{idx}}`;
                
                div.innerHTML = `
                    <img src="${{f.url}}" /> 
                    <div class="frame-info">Time: ${{f.timestamp_sec}}s</div> 
                    <button id="focus-btn-${{idx}}" class="focus-btn" onclick="toggleFrameFocus(${{idx}})">Focus / Select</button>
                `;
                container.appendChild(div);
            }});
        }}

        // --- Submit Judgement Function ---
        document.getElementById("submitSelected").onclick = async () => {{
            const selectedIndices = Array.from(selectedFrameIndices).sort((a, b) => a - b);
            if (selectedIndices.length === 0) {{ alert("Select at least one frame using the Focus button."); return; }}
            document.getElementById("serverResponse").innerHTML = "Sending frames for AI Judgement... Please wait.";


            const selectedURLs = selectedIndices.map(idx => {{
                const f = extractedFrames[idx];
                return f.url;
            }});

            const formData = new FormData();
            formData.append("figure_name", document.getElementById("figureSelect").value);
            formData.append("observations", document.getElementById("obsBox").value);
            formData.append("frame_urls_json", JSON.stringify(selectedURLs));

            const res = await fetch("/judge_frames", {{ method:"POST", body: formData }});
            const data = await res.json();
            
            const serverResponseDiv = document.getElementById("serverResponse");
            try {{
                serverResponseDiv.innerHTML = marked.parse(data.llm_output || "");
            }} catch (e) {{
                serverResponseDiv.textContent = JSON.stringify(data, null, 2);
                console.error("Error parsing LLM output:", e);
            }}
        }};
        
        // Ensure initial button state is checked when the page loads
        document.addEventListener('DOMContentLoaded', checkReadyState);
        </script>
    </div>
    </body>
    </html>
    """

# ------------------------
# Extract frames Endpoint (Optimized with 800px Resizing and Q75)
# ------------------------
@app.post("/extract_frames")
async def extract_frames(video: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define max width for resizing (Crucial for Base64 size reduction)
    MAX_WIDTH = 800 
    
    # Target number of frames to keep
    target_frames = 6
    # Calculate step size to cover the entire video (roughly 10 steps max)
    step = max(1, total_frames // 10) 

    frames = []
    count = 0
    while True: # Changed to 'while True' to process the entire video
        ret, frame = cap.read()
        if not ret: break # Exit loop when video ends

        if count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Only save frame if there is sufficient motion/detail
            if gray.std() >= 10:
                
                # --- RESIZE LOGIC ---
                height, width = frame.shape[:2]
                resized_frame = frame
                if width > MAX_WIDTH:
                    # Calculate new dimensions while maintaining aspect ratio
                    ratio = MAX_WIDTH / width
                    new_width = MAX_WIDTH
                    new_height = int(height * ratio)
                    
                    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                # --- END RESIZE LOGIC ---
                
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = os.path.join(FRAME_DIR, filename)
                
                # Save the frame using JPEG compression quality 75 (Aggressive size reduction)
                cv2.imwrite(filepath, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

                new_frame_data = {
                    "url": f"/frames/{filename}",
                    "timestamp_sec": round(count/fps,2)
                }
                
                # --- SLIDING WINDOW LOGIC (Keeps the latest frames) ---
                if len(frames) >= target_frames:
                    # Remove the OLDEST frame (index 0)
                    frames.pop(0) 
                
                # Add the NEW frame to the end
                frames.append(new_frame_data)
                # --- END SLIDING WINDOW LOGIC ---
                
        count += 1

    cap.release()
    os.remove(path)
    return {"frames": frames}


# ------------------------
# Judge frames with LLM Endpoint (REWRITTEN FOR GEMINI API)
# ------------------------
@app.post("/judge_frames")
async def judge_frames(
    figure_name: str = Form(...),
    observations: str = Form(""),
    frame_urls_json: str = Form(...)
):
    # Check if the Gemini client initialized successfully
    global client
    if not client:
        return JSONResponse(status_code=500, content={"llm_output": "Error: Gemini client not initialized. Check GEMINI_API_KEY."})
    
    frame_urls: List[str] = json.loads(frame_urls_json)

    # Load prompt template from JSON file
    try:
        with open("as_judging.json") as f:
            prompt_template = json.load(f)
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"llm_output": "Error: as_judging.json not found. Check if the file exists."})
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"llm_output": f"Error: Invalid JSON in as_judging.json: {e}"})

    # 1. Prepare text prompt
    gemini_content = []
    prompt_text = (
        f"Analyze the sequence of images for the Artistic Swimming Figure: '{figure_name}'. "
        f"Observations provided by the user: '{observations}'. "
        f"If the observations field is blank, base your assessment strictly on the visual evidence from the images. " 
        f"Reference the following judging guidelines: {prompt_template.get('content', 'No guidelines provided')}. "
        "Provide a technical assessment and a score out of 10. "
        "Format the response strictly using Markdown."
    )
    gemini_content.append(prompt_text)

    # 2. Iterate through URLs, read file locally, and create genai.types.Part objects
    files_processed = 0
    for url in frame_urls:
        # Convert the public URL path back to the local file path
        path = url.replace("/frames/", f"{FRAME_DIR}/")

        try:
            with open(path, "rb") as image_file:
                image_bytes = image_file.read()
                
                # Create the Gemini Part object
                image_part = genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg'
                )
                gemini_content.append(image_part)
                files_processed += 1
        except FileNotFoundError:
            print(f"File not found for Gemini encoding: {path}")
            continue 

    # --- Pre-API Check ---
    print(f"Sending {files_processed} images and prompt to Gemini...")
    if files_processed == 0 and len(frame_urls) > 0:
        return JSONResponse(status_code=400, content={"llm_output": "Error: Selected frames were not found on the server disk. Try re-extracting frames."})

    # --- GEMINI API Call & Response Handling ---
    output_text = ""
    try:
        completion = client.models.generate_content(
            model=MODEL_NAME, 
            contents=gemini_content,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=2000 
            )
        )
        
        output_text = completion.text
        
        # Check if the output is blank (failure or safety block)
        if not output_text:
            # Check the actual candidate list for a blockage reason
            if completion.candidates and completion.candidates[0].finish_reason.name == 'SAFETY':
                 # Extract and display the safety ratings
                 safety_details = "\n".join([
                     f"* **{r.category.name.replace('HARM_CATEGORY_', '').replace('_', ' ').title()}**: {r.probability.name}"
                     for r in completion.candidates[0].safety_ratings
                 ])
                 
                 print(f"WARNING: Gemini blocked the response due to safety filters.")
                 print(f"SAFETY REASON:\n{safety_details}")
                 
                 output_text = (
                     "## 🚨 Response Blocked by Safety Filters\n\n"
                     "The AI generated content that was flagged by safety filters. "
                     "This can happen with multimodal inputs. Try adjusting your prompt or selecting different frames. \n\n"
                     f"### Block Details:\n{safety_details}"
                 )
            else:
                # Catch other non-safety related blank responses
                print("WARNING: Gemini returned a blank response string for an unknown reason.")
                output_text = "The AI returned a blank response. Try running the process again."
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
        "num_frames": len(frame_urls),
        "figure_name": figure_name,
        "observations": observations
    }