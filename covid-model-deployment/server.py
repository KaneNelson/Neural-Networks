import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import predict


app_desc = """<h2>Upload image of lung X-ray with `predict`</h2>"""

app = FastAPI(title='Covid Detection Model', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be a .jpg or .jpeg image"
    image = await file.read()
    prediction = predict.predict(image)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app)