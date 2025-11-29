from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_videos(video1: UploadFile = File(...), video2: UploadFile = File(...)):
    path1 = os.path.join(UPLOAD_DIR, video1.filename)
    path2 = os.path.join(UPLOAD_DIR, video2.filename)

    with open(path1, "wb") as f1:
        f1.write(await video1.read())
    with open(path2, "wb") as f2:
        f2.write(await video2.read())

    # TODO: connect to your local LLM model here
    score = "8.7 / 10"
    comment = "Video 1 has smoother pacing; Video 2 has better lighting."

    return {"score": score, "comment": comment}