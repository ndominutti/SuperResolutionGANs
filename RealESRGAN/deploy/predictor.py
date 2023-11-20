from __future__ import print_function
import json
from fastapi import FastAPI, Response, UploadFile, Request, File
from fastapi.responses import JSONResponse, StreamingResponse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
import cv2
import base64
import io
from PIL import Image

model_path = "/opt/ml/model/net_g_latest.pth"


kwargs = {
            'tile':0,
            'tile_pad':10,
            'pre_pad':0,
            'fp32':False,
            'gpu_id':None,
            'outscale':4
        }


class ScoringService(object):
    model = None

    @classmethod
    def get_model(scoring_service, model_path, tile, tile_pad, pre_pad, fp32, gpu_id):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if scoring_service.model == None:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            scoring_service.model = RealESRGANer(
                                    scale=4,
                                    model_path=model_path,
                                    dni_weight=None,
                                    model=model,
                                    tile=tile,
                                    tile_pad=tile_pad,
                                    pre_pad=int(pre_pad),
                                    half=not fp32,
                                    gpu_id=gpu_id
            )
            
    @classmethod
    def preprocess(scoring_service, np_image):
        """
        """
        im2 = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
        return im2    
        
        
    @classmethod
    def predict(scoring_service, np_image, tile, tile_pad, pre_pad, fp32,
                 gpu_id, outscale):
        """For the input, do the predictions and return them.

        Args:
            input (pd.DataFrame): The data on which to do the predictions. There will be
                one prediction per row in the dataframe

        Returns:
            _type_: _description_
        """
        print('Upscaling...')
        scoring_service.get_model(model_path, tile, tile_pad, pre_pad, fp32, gpu_id)
        img  = scoring_service.preprocess(np_image)
        try:
            output, _ = scoring_service.model.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        return output



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
    """Do an inference on a single image. 
    """
    if request.headers.get("Content-Type").startswith("image/"):
        request_body = await request.body() 
        image_data = io.BytesIO(request_body) 
        np_image = np.frombuffer(image_data.getvalue(), np.uint8)
    else:
        return Response(
            content="This predictor only supports Images data (image/png, image/jpeg)",
            status_code=415,
            media_type="text/plain",
        )
    upscaled_img = ScoringService.predict(np_image, **kwargs)
    _, img_encoded = cv2.imencode(".png", upscaled_img)
    return StreamingResponse(io.BytesIO(img_encoded), media_type="image/png")
