from typing import List

from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tfidf import tfidf


app = FastAPI()
templates = Jinja2Templates(directory="static")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)) -> Response:
    try:
        contents = []
        for file in files:
            with file.file as f:
                content = f.read()
                contents.append(content)

        res = tfidf(contents)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong {e}")

    context = {
        "request": request,
        "data": res,
        "columns": ["слово", "tf", "idf"],
    }
    return templates.TemplateResponse("results.html", context)


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
