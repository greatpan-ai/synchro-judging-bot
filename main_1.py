import os
import cv2
import tempfile
import json
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

# ------------------------
# Config
# ------------------------
# NOTE: Replace with your actual model name if different
client = OpenAI()
MODEL_NAME = "gpt-4.1-mini" 
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
# Extract frames Endpoint
# ------------------------
@app.post("/extract_frames")
async def extract_frames(video: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # UPDATED: Changed from 5 to 6 target frames
    target_frames = 6
    step = max(1, total_frames // (target_frames*2)) 

    frames = []
    count = 0
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret: break

        if count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.std() >= 10:
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = os.path.join(FRAME_DIR, filename)
                cv2.imwrite(filepath, frame)

                frames.append({
                    "url": f"/frames/{filename}",
                    "timestamp_sec": round(count/fps,2)
                })
        count += 1

    cap.release()
    os.remove(path)
    return {"frames": frames}


# ------------------------
# Judge frames with LLM Endpoint
# ------------------------
@app.post("/judge_frames")
async def judge_frames(
    figure_name: str = Form(...),
    observations: str = Form(""),
    frame_urls_json: str = Form(...)
):
    frame_urls = json.loads(frame_urls_json)

    try:
        with open("as_judging.json") as f:
            prompt_template = json.load(f)
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"llm_output": "Error: as_judging.json not found."})
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"llm_output": f"Error: Invalid JSON in as_judging.json: {e}"})

    llm_input_data = {
        "role": "user",
        "figure_name": figure_name,
        "observations": observations,
        "key_frames": [{"url": url} for url in frame_urls],
        "system_prompt_content": prompt_template.get("content", []) 
    }

    try:
        completion = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(llm_input_data)
                        }
                    ]
                }
            ]
        )
        output_text = completion.output_text
    except Exception as e:
        output_text = f"LLM call failed: {type(e).__name__}: {e}"

    return {
        "llm_output": output_text,
        "num_frames": len(frame_urls),
        "figure_name": figure_name,
        "observations": observations
    }