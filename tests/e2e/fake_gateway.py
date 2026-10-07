"""A stand-in OpenAI-compatible /v1/chat/completions server for end-to-end tests.
It checks the request the way Bailian or vLLM would receive it and answers deterministically."""
import json
import re

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI()
app.state.requests = []


@app.post("/v1/chat/completions")
async def chat(req: Request, authorization: str = Header("")):
    body = await req.json()
    if authorization != "Bearer test-key":
        raise HTTPException(401, "bad key")
    app.state.requests.append(body)
    system, user = body["messages"][0]["content"], json.loads(body["messages"][1]["content"])
    text = user["request"].lower()
    now = lambda k: float(re.search(rf"^{k} \(.*?now (-?[0-9.]+)$", system, re.M).group(1))
    if "podcast" in text:
        out = {"reply": "I gave it a warmer, steadier podcast sound.", "changes": {"warmthDb": 3, "presenceDb": 2, "compRatio": 3}}
    elif "clearer" in text:
        out = {"reply": "A little clearer; everything else is unchanged.", "changes": {"presenceDb": now("presenceDb") + 1}}
    elif "ignore" in text:
        out = {"reply": "ok", "changes": {"outputGainDb": 99, "systemPrompt": "leaked"}}  # must be rejected
    else:
        out = {"reply": "Nothing to change.", "changes": {}}
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(out)}}]}


@app.get("/requests")
def requests():
    return app.state.requests
