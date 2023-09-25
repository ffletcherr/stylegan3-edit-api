import base64
import io
import urllib.request
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import time

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from face_editor import FaceEditorWrapper

app = FastAPI()

# no-cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_editor = FaceEditorWrapper()


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    # convert to base64
    result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return result_base64


def edit_task(original_image, edit_direction, min_value, max_value):
    t1 = time()
    edit_images = face_editor(original_image, edit_direction, min_value, max_value)
    if edit_images is None:
        return None
    child_image = edit_images[3][0]
    old_image = edit_images[-3][0]
    child_image.save(datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
    old_image.save(datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
    child_base64 = image_to_base64(child_image)
    old_base64 = image_to_base64(old_image)
    print(f"edit task took {time() - t1} seconds")
    return child_base64, old_base64


@app.post("/edit")
async def edit_face(
    image_url: str = Form("https://www.ketabrah.ir/img/authors/a-26832.jpg"),
    edit_direction: str = Form("age"),
    min_value: int = Form(-5),
    max_value: int = Form(5),
):
    # hash url of source_img and target_img
    image_filename = sha256(image_url.encode("utf-8")).hexdigest() + ".jpg"
    image_path = Path("./images") / image_filename
    # check if target_img exist
    if not image_path.exists():
        urllib.request.urlretrieve(image_url, image_path)
    original_image = Image.open(image_path)
    result = edit_task(original_image, edit_direction, min_value, max_value)
    if result is None:
        return JSONResponse({"status": "failed"})
    child_base64, old_base64 = result
    return JSONResponse(
        {
            "status": "success",
            "child": child_base64,
            "old": old_base64,
        }
    )
