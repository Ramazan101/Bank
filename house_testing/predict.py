import uvicorn
from fastapi import FastAPI
import main2

house_app = FastAPI()
house_app.include_router(main2.predict_router)


if __name__ == "__main__":
    uvicorn.run(house_app, host="127.0.0.1", port=8002)
