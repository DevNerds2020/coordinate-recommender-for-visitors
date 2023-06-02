from fastapi import FastAPI 


app = FastAPI()

@app.post("/recommend")
async def request_recommendation():
    r = Recommendation()
    c = r.read_file().iloc[:100, 2:4]
    c.columns = [f'C{n}' for n in range(1, 3)]
    print(c.head())
    c = c.dropna()
    model = r.Agglomerative_CompleteLink_cluster(c, 10)
    r.show_plt(model, c, 'Agglomerative Complete Link')
    return {"message": "Hello World"}