from fastapi import FastAPI
import uvicorn
from Avocado.api import predict

avocado_app = FastAPI(title='Avocado ML')
avocado_app.include_router(predict.avocado_predict)


if __name__ == '__main__':
    uvicorn.run(avocado_app, host='127.0.0.1', port=8000)