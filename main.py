# coding: utf-8
import base64
import io
import operator
import os
import tempfile
import time
from typing import List

import uvicorn
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.responses import HTMLResponse

from app.api_models import Face, NearestNeighbour
from app.face_embedder import FaceEmbedder
from app.indexer import FaissIndexer
from app.utils.blobstorage_helper import BlobStorageHelper

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Get environment from env variable
environment = os.environ["ENVIRONMENT"] if ("ENVIRONMENT" in os.environ) else "development"
print("ENVIRONMENT : {}".format(environment))

# Load configuration from ENVIRONMENT variable
with open("./config/{}.yml".format(environment), "r") as configuration_file:
    conf = yaml.load(configuration_file, Loader=yaml.FullLoader)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load face embedder 1st
@app.on_event('startup')
def startup_event():
    connection_string = conf["connection_string"]
    global face_embedder, face_embedder_nn, faiss_indexer, reference_imgs, blob_helper
    blob_helper = BlobStorageHelper(account_name="ovlabsearch",
                                    connection_string=connection_string)

    faiss_indexer = FaissIndexer()
    index_data = blob_helper.get_file_as_bytes(container_name="index",
                                               remote_file_name="face-ov/test_faiss_index_2")
    faiss_indexer.load_bytes(index_data)

    face_embedder = FaceEmbedder(post_process=False, keep_all=True, selection_method="probability")
    face_embedder_nn = FaceEmbedder()

    reference_imgs = blob_helper.get_file_as_json(container_name="index",
                                                  remote_file_name="face-ov/test_indexed_filenames_2.txt")
    reference_imgs = [img.replace("/dbfs/mnt/ovlabdata/", "") for img in reference_imgs]


@app.get("/")
def read_root():
    return {"Welcome to ": "OV Lab"}


@app.post("/get-faces", response_model=List[Face])
async def get_faces(file: UploadFile = File(...)):
    img = Image.open(file.file)
    faces = face_embedder.crop_image(img, return_type="list", permute=True)

    return faces


@app.post("/get-faces-2/")
async def get_faces_2(file: UploadFile = File(...), index: int = Form(...)):
    img = Image.open(file.file)
    # Permute when we want numpy type
    faces = face_embedder.crop_image(img, return_type="np", permute=True)

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as outfile:
        Image.fromarray(faces[index]["face"], 'RGB').save(outfile)
        return FileResponse(outfile.name, media_type="image/png")


@app.post("/get-nearest-neighbour/", response_model=List[NearestNeighbour])
async def get_nearest_neighbour(file: UploadFile = File(...), limit: int = Form(...)):
    img = Image.open(file.file)
    embeddings, _ = face_embedder_nn.crop_and_embed(img, return_type="np", type=None, normalize=True)
    # Batch search here as we can have several faces in one picture
    results = faiss_indexer.batch_search(embeddings, limit=limit)

    return [{"nearest_neighbours_path": [reference_imgs[r[0]] for r in res],
             "nearest_neighbours_distance": [r[1] for r in res],
             "predicted_identity": "toto"} for res in results]


@app.get("/test-get-nearest-neighbour/")
async def test_get_nearest_neighbour(img_as_str: str, limit: int, thresh: float):
    myfile_as_str = img_as_str.split(',')
    imgdata = base64.b64decode(myfile_as_str[1])
    im = Image.open(io.BytesIO(imgdata))
    embeddings, _ = face_embedder_nn.crop_and_embed(im, return_type="np", type=None, normalize=True)
    # Batch search here as we can have several faces in one picture
    results = faiss_indexer.batch_search(embeddings, limit=limit)

    results = [{"nearest_neighbours_path": reference_imgs[r[0]],
                "nearest_neighbours_distance": r[1],
                "nearest_neighbours_name": reference_imgs[r[0]].split('/')[-2]} for r in results[0]]

    list_img_data = []

    for element in results:

        if element['nearest_neighbours_distance'] <= thresh:
            remote_filename = element["nearest_neighbours_path"]
            celebrity_name = element["nearest_neighbours_name"]
            img_data = blob_helper.get_file_as_bytes(container_name="data",
                                                     remote_file_name=remote_filename)

            list_img_data.append([base64.b64encode(img_data), remote_filename, celebrity_name])

    celebrity_accumulator = {}
    for celebrity in results:
        if not celebrity["nearest_neighbours_name"] in celebrity_accumulator:
            celebrity_accumulator[celebrity["nearest_neighbours_name"]] = 1
        else:
            celebrity_accumulator[celebrity["nearest_neighbours_name"]] += 1

    max_celebrity_accumulator = max(celebrity_accumulator.items(), key=operator.itemgetter(1))

    if len(list_img_data) == 0:
        return [list_img_data, -1, 0]
    else:
        return [list_img_data, max_celebrity_accumulator[0], max_celebrity_accumulator[1]]


@app.get("/extract-faces")
async def get_crop_faces():
    content = """
       <html>
       <body>
       </form>
           <form action="/get-faces/" enctype="multipart/form-data" method="post">
           <input type="file" name="file"></label>
           <input type="submit" value="Go">
       </form>
       </body>
       </html>
   """
    return HTMLResponse(content=content)


@app.get("/extract-faces-2")
async def get_crop_faces():
    content = """
       <html>
       <body>
       </form>
           <form action="/get-faces-2/" enctype="multipart/form-data" method="post">
           <p>
           <input type="file" name="file" ></label>
           </p>
           <p>
           <input type="number" id="index" name="index" size="30">
           </p>
        <div>
        <input type="submit" value="Send">
        </div>
       </form>
       </body>
       </html>
   """
    return HTMLResponse(content=content)


@app.get("/photo")
async def read_index():
    return FileResponse('static/photoclick.html')


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == '__main__':
    uvicorn.run("main:app")
