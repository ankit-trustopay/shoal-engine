from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Shoal AI Engine", version="0.1.0")


class IgniteRequest(BaseModel):
    swarmId: str = Field(..., min_length=1)
    premise: str = Field(..., min_length=1)


class IgniteResponse(BaseModel):
    status: str
    swarmId: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
def ignite(payload: IgniteRequest) -> IgniteResponse:
    return IgniteResponse(status="Swarm ignited", swarmId=payload.swarmId)
