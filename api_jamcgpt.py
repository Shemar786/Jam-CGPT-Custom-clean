from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/generate-sql")
def generate_sql(request: QueryRequest):
    try:
        prompt = request.question
        # Call your script using subprocess
        result = subprocess.run(
            ["python3", "sample_jam_cgpt.py", "--prompt", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        sql = result.stdout.strip().split('\n')[-1]  # Get last line of output
        return {"sql": sql}
    except Exception as e:
        return {"error": str(e)}
