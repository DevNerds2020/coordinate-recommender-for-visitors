from fastapi import FastAPI 


app = FastAPI()

@app.post("/recommend")
async def request_recommendation():
    return
