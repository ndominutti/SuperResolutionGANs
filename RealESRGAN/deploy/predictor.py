from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import json
from fastapi import FastAPI, Response, UploadFile, Request
from fastapi.responses import JSONResponse
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, "decision-tree-model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (pd.DataFrame): The data on which to do the predictions. There will be
                one prediction per row in the dataframe

        Returns:
            _type_: _description_
        """
        clf = cls.get_model()
        return clf.predict(input)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Working": "API"}


@app.get("/ping")
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None

    status  = 200 if health else 404
    content = {"status":"Healthy container" if health else "Resource not found"}
    json_content = json.dumps(content)
    return JSONResponse(content=json_content)


@app.post("/invocations")
async def predict(request: Request):
    """Do an inference on a single batch of data. 
    """
    dataset = None

    if request.headers.get("Content-Type") == "text/csv": 
        request_body = await request.body()
        request_body_str = request_body.decode('utf-8') 
        s = io.StringIO(request_body_str)
        dataset = pd.read_csv(s, header=None)
    else:
        return Response(
            content="This predictor only supports CSV data",
            status_code=415,
            media_type="text/plain",
        )
    
    predictions = ScoringService.predict(dataset)
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    return Response(content=result, status_code=200, media_type="text/csv")