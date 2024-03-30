from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tfidf import tfidf


app = FastAPI()
templates = Jinja2Templates(directory="static")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> Response:
    try:
        content = file.file.read()
        res = tfidf([content])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong {e}")

    finally:
        file.close()

    context = {
        "request": request,
        "data": res,
        "columns": ["слово", "tf", "idf"],
    }
    return templates.TemplateResponse("results.html", context)


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
