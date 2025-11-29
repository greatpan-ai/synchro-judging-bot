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
            /* UPDATE 3: Change all primary buttons to blue (#007bff) */
            button {{
                background-color: #007bff; /* Primary Blue */
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
                margin-left: 10px;
            }}
            button:hover {{
                background-color: #0056b3; /* Darker Blue */
            }}
            /* Disabled button style */
            button[disabled] {{
                background-color: #cccccc !important; 
                cursor: not-allowed;
            }}

            /* CSS for the image header */
            #headerLogo {{
                width: 100%;
                max-height: 200px;
                height: auto;
                object-fit: fill;
                display: block; 
                margin: 0 auto 20px auto; 
            }}


            /* FRAME STYLES (3-COLUMN GRID) */
            #framesContainer {{
                display: grid; 
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
                background-color: #dc3545; /* Red for selected state */
            }}
            
            /* VERTICAL COMPACTION */
            .control-group {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 20px;
            }}
            /* Specific style for the checkbox/label group */
            #sampleGroup {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}


            /* AI JUDGEMENT RESPONSE STYLES */
            #serverResponse {{
                border: 2px solid #ffc107;
                padding: 15px; 
                background: #fffbe6;
                border-radius: 6px;
                font-size: 0.95em;
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
            #attribution {{ /* Style for the new attribution text (Update 4) */
                font-size: 0.75em;
                color: #6c757d;
                text-align: center;
                margin-top: 20px;
                padding-top: 10px;
                border-top: 1px solid #e9ecef;
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

        <div class="control-group" style="align-items: flex-start; flex-direction: column;">
            
            <div id="videoSelectionGroup" class="control-group">
                <div id="sampleGroup">
                    <input type="checkbox" id="useSampleCheck">
                    <label for="useSampleCheck">Use Sample Video</label>
                </div>

                <input type="file" id="videoInput" accept="video/*" style="margin-bottom: 0; width: auto;">
            </div>

            <button id="processBtn" disabled style="margin-left: 0; margin-top: 10px; width: 300px;">
                Process Video
            </button>
        </div>


        <hr/>
        <h2>Select Key Frames 📸</h2>
        <div id="framesContainer"></div>

        <hr/>
        <h2>Select Figure 📝</h2>
        
        <h3>Select Figure & Judge</h3>
        <div class="control-group">
            <select id="figureSelect">
                <optgroup label="100 Series – Novice / Basic Figures">
                    <option value="101 Front Layout">101 Front Layout</option>
                    <option value="102 Back Layout">102 Back Layout</option>
                    <option value="103 Front Pike">103 Front Pike</option>
                    <option value="104 Back Pike">104 Back Pike</option>
                    <option value="105 Front Layout to Ballet Leg">105 Front Layout to Ballet Leg</option>
                    <option value="106 Back Layout to Ballet Leg">106 Back Layout to Ballet Leg</option>
                    <option value="107 Front Layout to Split">107 Front Layout to Split</option>
                    <option value="108 Back Layout to Split">108 Back Layout to Split</option>
                    <option value="109 Front Layout to Vertical">109 Front Layout to Vertical</option>
                </optgroup>
                <optgroup label="200 Series – Intermediate Figures">
                    <option value="201 Front Pike to Vertical">201 Front Pike to Vertical</option>
                    <option value="202 Back Pike to Vertical">202 Back Pike to Vertical</option>
                    <option value="203 Front Layout to Flamingo">203 Front Layout to Flamingo</option>
                    <option value="204 Back Layout to Flamingo">204 Back Layout to Flamingo</option>
                    <option value="205 Front Layout to Tower">205 Front Layout to Tower</option>
                    <option value="206 Back Layout to Tower">206 Back Layout to Tower</option>
                    <option value="207 Front Layout to Catalina">207 Front Layout to Catalina</option>
                    <option value="208 Back Layout to Catalina">208 Back Layout to Catalina</option>
                    <option value="209 Front Layout to Kip">209 Front Layout to Kip</option>
                </optgroup>
                <optgroup label="300 Series – Advanced / Youth & Junior Figures">
                    <option value="301 Barracuda">301 Barracuda</option>
                    <option value="302 Barracuda Airborne Split">302 Barracuda Airborne Split</option>
                    <option value="303 Barracuda Twist">303 Barracuda Twist</option>
                    <option value="304 Barracuda Vertical Descent">304 Barracuda Vertical Descent</option>
                    <option value="305 Kip">305 Kip</option>
                    <option value="306 Kip Twist">306 Kip Twist</option>
                    <option value="307 Kip Vertical Descent">307 Kip Vertical Descent</option>
                    <option value="308 Flamingo">308 Flamingo</option>
                    <option value="309 Flamingo Bent Knee">309 Flamingo Bent Knee</option>
                    <option value="310 Flamingo Bent Knee to Vertical">310 Flamingo Bent Knee to Vertical</option>
                    <option value="311 Flamingo Vertical Descent">311 Flamingo Vertical Descent</option>
                    <option value="312 Tower">312 Tower</option>
                    <option value="313 Tower Split">313 Tower Split</option>
                    <option value="314 Tower Split to Vertical">314 Tower Split to Vertical</option>
                    <option value="315 Tower Vertical Descent">315 Tower Vertical Descent</option>
                    <option value="316 Catalina">316 Catalina</option>
                    <option value="317 Walkover Front">317 Walkover Front</option>
                    <option value="318 Walkover Back">318 Walkover Back</option>
                    <option value="319 Walkover Front to Split">319 Walkover Front to Split</option>
                    <option value="320 Walkover Back to Split">320 Walkover Back to Split</option>
                    <option value="321 Walkover Front to Vertical Split">321 Walkover Front to Vertical Split</option>
                    <option value="322 Walkover Back to Vertical Split">322 Walkover Back to Vertical Split</option>
                    <option value="323 Walkover Front to Vertical Descent">323 Walkover Front to Vertical Descent</option>
                    <option value="324 Walkover Back to Vertical Descent">324 Walkover Back to Vertical Descent</option>
                    <option value="325 Airborne Split">325 Airborne Split</option>
                    <option value="326 Airborne Split to Vertical">326 Airborne Split to Vertical</option>
                    <option value="327 Airborne Split to Vertical Descent">327 Airborne Split to Vertical Descent</option>
                    <option value="328 Kip Twist to Vertical">328 Kip Twist to Vertical</option>
                    <option value="329 Kip Twist to Vertical Descent">329 Kip Twist to Vertical Descent</option>
                    <option value="330 Flamingo Bent Knee to Vertical Descent">330 Flamingo Bent Knee to Vertical Descent</option>
                    <option value="331 Tower Split to Vertical Descent">331 Tower Split to Vertical Descent</option>
                    <option value="332 Barracuda Airborne Split to Vertical">332 Barracuda Airborne Split to Vertical</option>
                    <option value="333 Barracuda Airborne Split to Vertical Descent">333 Barracuda Airborne Split to Vertical Descent</option>
                    <option value="334 Barracuda Twist to Vertical">334 Barracuda Twist to Vertical</option>
                    <option value="335 Barracuda Twist to Vertical Descent">335 Barracuda Twist to Vertical Descent</option>
                    <option value="336 Kip Twist to Vertical Split">336 Kip Twist to Vertical Split</option>
                    <option value="337 Kip Twist to Vertical Split Descent">337 Kip Twist to Vertical Split Descent</option>
                    <option value="338 Flamingo Bent Knee to Vertical Split">338 Flamingo Bent Knee to Vertical Split</option>
                    <option value="339 Flamingo Bent Knee to Vertical Split Descent">339 Flamingo Bent Knee to Vertical Split Descent</option>
                    <option value="340 Catalina Walkover">340 Catalina Walkover</option>
                    <option value="341 Catalina Walkover to Split">341 Catalina Walkover to Split</option>
                    <option value="342 Catalina Walkover to Vertical Split">342 Catalina Walkover to Vertical Split</option>
                    <option value="343 Catalina Walkover to Vertical Descent">343 Catalina Walkover to Vertical Descent</option>
                    <option value="344 Tower Walkover">344 Tower Walkover</option>
                    <option value="345 Tower Walkover to Split">345 Tower Walkover to Split</option>
                    <option value="346 Tower Walkover to Vertical Split">346 Tower Walkover to Vertical Split</option>
                    <option value="347 Tower Walkover to Vertical Descent">347 Tower Walkover to Vertical Descent</option>
                    <option value="348 Flamingo Walkover">348 Flamingo Walkover</option>
                    <option value="349 Flamingo Walkover to Vertical">349 Flamingo Walkover to Vertical</option>
                </optgroup>
                <optgroup label="400 Series – Senior / High Difficulty Figures">
                    <option value="401 Flamingo Bent Knee to Vertical Descent">401 Flamingo Bent Knee to Vertical Descent</option>
                    <option value="402 Flamingo Bent Knee to Vertical Split Descent">402 Flamingo Bent Knee to Vertical Split Descent</option>
                    <option value="403 Tower Split to Vertical Descent">403 Tower Split to Vertical Descent</option>
                    <option value="404 Tower Walkover to Vertical Split Descent">404 Tower Walkover to Vertical Split Descent</option>
                    <option value="405 Barracuda Airborne Split to Vertical">405 Barracuda Airborne Split to Vertical</option>
                    <option value="406 Barracuda Airborne Split to Vertical Descent">406 Barracuda Airborne Split to Vertical Descent</option>
                    <option value="407 Barracuda Twist to Vertical">407 Barracuda Twist to Vertical</option>
                    <option value="408 Barracuda Twist to Vertical Descent">408 Barracuda Twist to Vertical Descent</option>
                    <option value="409 Kip Twist to Vertical">409 Kip Twist to Vertical</option>
                    <option value="410 Kip Twist to Vertical Descent">410 Kip Twist to Vertical Descent</option>
                    <option value="411 Kip Twist to Vertical Split">411 Kip Twist to Vertical Split</option>
                    <option value="412 Kip Twist to Vertical Split Descent">412 Kip Twist to Vertical Split Descent</option>
                    <option value="413 Flamingo Walkover to Vertical Split">413 Flamingo Walkover to Vertical Split</option>
                    <option value="414 Flamingo Walkover to Vertical Split Descent">414 Flamingo Walkover to Vertical Split Descent</option>
                    <option value="415 Tower Walkover to Vertical Split">415 Tower Walkover to Vertical Split</option>
                    <option value="416 Tower Walkover to Vertical Split Descent">416 Tower Walkover to Vertical Split Descent</option>
                    <option value="417 Catalina Walkover to Vertical Split">417 Catalina Walkover to Vertical Split</option>
                    <option value="418 Catalina Walkover to Vertical Split Descent">418 Catalina Walkover to Vertical Split Descent</option>
                    <option value="419 Airborne Split to Vertical Split">419 Airborne Split to Vertical Split</option>
                    <option value="420 Airborne Split to Vertical Split Descent">420 Airborne Split to Vertical Split Descent</option>
                </optgroup>
                <option value="Other">Other</option>
            </select>
            
            <button id="submitSelected" style="margin-top:0;">Send to Judging Bot</button>
        </div>

        <hr/>
        <h2>Judge 🤖</h2>
        <div id="serverResponse" style="min-height: 100px;">Awaiting Judgement...</div>

        <div id="attribution">Figure Judging Bot V1.0 was developed using ChatGPT 5.1 and trained on the Artistic Swimming Manual for Judges, Technical Controllers, Referees, and Coaches (2022–2025).</div>

        <script>
        // *** VARIABLE DEFINITIONS ***
        let extractedFrames = [];
        let videoFileToProcess = null; // Holds the File object for upload or sample
        let selectedFrameIndices = new Set();
        const videoInput = document.getElementById("videoInput");
        const useSampleCheck = document.getElementById("useSampleCheck");
        const processBtn = document.getElementById("processBtn");
        const serverResponse = document.getElementById("serverResponse");


        // --- CORE FUNCTION: Checks state and enables the Process button ---
        function checkReadyState() {{
            const fileSelected = videoInput.files.length > 0;
            const useSample = useSampleCheck.checked;
            
            // Enable button if a file is selected OR the sample box is checked
            processBtn.disabled = !(fileSelected || useSample);
            
            // Disable file input if sample is checked
            videoInput.disabled = useSample;
            
            // Reset file input value if sample is checked
            if (useSample) {{
                videoInput.value = '';
            }}
        }}

        // --- CORE FUNCTION: Extracts frames and renders them ---
        async function runFrameExtraction(fileToProcess) {{
            if (!fileToProcess) {{ 
                serverResponse.innerHTML = "Error: No video file is selected or loaded.";
                return; 
            }}
            
            // ** START LOADING STATE **
            processBtn.disabled = true;
            processBtn.textContent = 'Extracting key frames...'; // Update button text
            serverResponse.innerHTML = "<h3>⏳ Extracting key frames from video... please wait.</h3>";

            const formData = new FormData();
            formData.append("video", fileToProcess);

            try {{
                const res = await fetch("/extract_frames", {{ method:"POST", body: formData }});
                const data = await res.json();

                // ** END LOADING STATE **
                processBtn.disabled = false;
                processBtn.textContent = 'Process Video'; // Restore button text

                extractedFrames = data.frames || [];
                selectedFrameIndices.clear(); 
                renderFrames();
                serverResponse.innerHTML = "Frames extracted. Select key frames using the 'Select' button and press 'Send to Judging Bot' to continue.";
            }} catch (error) {{
                processBtn.disabled = false;
                processBtn.textContent = 'Process Video'; // Restore button text
                serverResponse.innerHTML = `<h3>❌ Error during frame extraction: ${{error.message}}</h3>`;
                console.error(error);
            }}
        }}

        // --- Event Listener for Checkbox/Input changes ---
        videoInput.onchange = checkReadyState;
        useSampleCheck.onchange = checkReadyState;

        // --- Event Listener for Unified Process Button ---
        processBtn.onclick = async () => {{
            const useSample = useSampleCheck.checked;
            let fileToProcess = null;

            if (useSample) {{
                // Logic for fetching the sample video
                serverResponse.innerHTML = "Loading sample video file...";
                try {{
                    const response = await fetch('{SAMPLE_VIDEO_PATH}');
                    const blob = await response.blob();
                    fileToProcess = new File([blob], "sample_video.mp4", {{type: "video/mp4"}});
                }} catch (e) {{
                    serverResponse.innerHTML = `Error loading sample video. Ensure the file exists.`;
                    console.error("Error fetching sample video:", e);
                    checkReadyState();
                    return;
                }}
            }} else if (videoInput.files.length > 0) {{
                // Logic for uploaded video
                fileToProcess = videoInput.files[0];
            }}

            if (fileToProcess) {{
                await runFrameExtraction(fileToProcess);
            }} else {{
                serverResponse.innerHTML = "Please select a file or check 'Use Sample Video'.";
            }}
        }};
        
        // --- Render Frames Function ---
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
                    <button id="focus-btn-${{idx}}" class="focus-btn" onclick="toggleFrameFocus(${{idx}})">Select</button>
                `;
                container.appendChild(div);
            }});
        }}
        
        function toggleFrameFocus(frameIndex) {{
            const frameBox = document.getElementById(`frame-box-${{frameIndex}}`);
            const focusBtn = document.getElementById(`focus-btn-${{frameIndex}}`);

            if (selectedFrameIndices.has(frameIndex)) {{
                selectedFrameIndices.delete(frameIndex);
                frameBox.classList.remove('focused');
                focusBtn.classList.remove('selected');
                focusBtn.textContent = 'Select'; 
            }} else {{
                selectedFrameIndices.add(frameIndex);
                frameBox.classList.add('focused');
                focusBtn.classList.add('selected');
                focusBtn.textContent = 'Selected';
            }}
        }}


        // --- Submit Judgement Function ---
        document.getElementById("submitSelected").onclick = async () => {{
            const selectedIndices = Array.from(selectedFrameIndices).sort((a, b) => a - b);
            const submitBtn = document.getElementById("submitSelected");
            const serverResponseDiv = document.getElementById("serverResponse");

            if (selectedIndices.length === 0) {{ alert("Select at least one frame using the Select button."); return; }}
            
            // ** START LOADING STATE **
            submitBtn.disabled = true;
            submitBtn.textContent = 'AI Judging in Progress...';
            serverResponseDiv.innerHTML = "<h3>🤖 Sending frames to Judging Bot for Judgement... Please wait.</h3>"; 

            const selectedBase64Data = selectedIndices.map(idx => {{
                return extractedFrames[idx].base64_data;
            }});

            const formData = new FormData();
            formData.append("figure_name", document.getElementById("figureSelect").value);
            formData.append("frame_base64_json", JSON.stringify(selectedBase64Data));

            try {{
                const res = await fetch("/judge_base64_frames", {{ method:"POST", body: formData }});
                const data = await res.json();
                
                // ** END LOADING STATE **
                submitBtn.disabled = false;
                submitBtn.textContent = 'Send to Judging Bot'; 

                try {{
                    serverResponseDiv.innerHTML = marked.parse(data.llm_output || "");
                }} catch (e) {{
                    serverResponseDiv.textContent = JSON.stringify(data, null, 2);
                    console.error("Error parsing LLM output:", e);
                }}
            }} catch (error) {{
                // ** HANDLE ERROR **
                submitBtn.disabled = false;
                submitBtn.textContent = 'Send to Judging Bot'; 
                serverResponseDiv.innerHTML = `<h3>❌ Network Error: Could not reach the server.</h3>`;
                console.error("Fetch error during judgement submission:", error);
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