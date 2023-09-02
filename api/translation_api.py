import time
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# Add the directories to sys.path
sys.path.extend([script_dir, parent_dir, grand_dir])
from common.model_types import ModelTypes
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.config import Configuration
from pipeline.full_pipeline import TranslationPipeline

from starlette.background import BackgroundTask

class TranslationItem(BaseModel):
    text: str
    model: str


def create_app():
    config = Configuration()
    pipeline = TranslationPipeline(config)
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/translate/text")
    async def translateText(translation: TranslationItem):
        text = translation.text
        model = translation.model
        translated_text, model_type = await pipeline(text, model)
        return {
            'IsSuccessed': True,
            'Message': 'Success',
            'ResultObj': {
                'src': text.split('\n'),
                'tgt': translated_text
            }
        }

    @app.post("/translate/file")
    async def translateFile(file: UploadFile = File(...), model: str = Form(...)):
        content = await file.read()
        text = content.decode('utf-8')
        translated_text, model_type = await pipeline(text, model)

        tmp_dir = 'tmp_files'
        isExist = os.path.exists(tmp_dir)
        if not isExist: 
            os.makedirs(tmp_dir)
            print(f'Directory "{tmp_dir}" is created!')

        filename = str(int(time.time())) + file.filename
        with open(f'{tmp_dir}/{filename}', encoding='utf-8', mode='w') as f:
            f.write('\n'.join(translated_text))
        
        def cleanup(filename):
            os.remove(f'{tmp_dir}/{filename}')

        return FileResponse(
            f'{tmp_dir}/{filename}',
            background=BackgroundTask(cleanup, filename),
        )

    @app.get("/models")
    async def getModels():
        return {
            'models': ModelTypes.get_models()
        }

    return app


if __name__ == "__main__":
    app_ = create_app()
    uvicorn.run(app_, host="0.0.0.0", port=8000)


