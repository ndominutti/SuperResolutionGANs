from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, Response, Request
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
import utils
import json
import cv2
import io


config_dict = utils.load_config('/opt/program/config/config.yml')
MODEL_PATH  = config_dict['model_path']
KWARGS      = config_dict['model_params']


class UpscalingService(object):
    """
    Create a upscaling service with a RealESRGANer model (https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py) under the hood.
    """
    model = None

    @classmethod
    def get_model(upscaler_service, model_path:str, tile:int, tile_pad:int, pre_pad:int, fp32:bool, device:str) -> None:
        """
        Get the model object for this instance, loading it if it's not already loaded.

        Args:
            model_path(str): Path to the model's weights
            tile (int)     : As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
            tile_pad (int) : The pad size for each tile, to remove border artifacts. Default: 10.
            pre_pad (int)  : Pad the input images to avoid border artifacts. Default: 10.
            fp32(bool)     : If set to false, use half precision during inference (only available for cuda device).
            device(str)    : Define the device to use

        Returns 
            None
        """
        if upscaler_service.model == None:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upscaler_service.model = RealESRGANer(
                                    scale=4,
                                    model_path=model_path,
                                    dni_weight=None,
                                    model=model,
                                    tile=tile,
                                    tile_pad=tile_pad,
                                    pre_pad=int(pre_pad),
                                    half=not fp32,
                                    gpu_id=None,
                                    device=device
            )
            
    @classmethod
    def preprocess(upscaler_service, np_image:np.ndarray) -> np.ndarray:
        """
        Apply minor preprocessing to the input image.

        Args:
            np_image(np.array): image in form of a numpy array

        Returns
            np.ndarray: decoded image
        """
        im2 = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
        return im2    
        
    @classmethod
    def predict(upscaler_service, np_image:np.ndarray, tile:0, tile_pad:0, pre_pad:0, fp32:bool,
                 outscale:int, device:str) -> np.ndarray:
        """For the input, preprocess them and then do the predictions and return them.

        Args:
            np_image(np.ndarray): Input image in the form of a numpy ndarray
            tile (int)          : As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
            tile_pad (int)      : The pad size for each tile, to remove border artifacts. Default: 10.
            pre_pad (int)       : Pad the input images to avoid border artifacts. Default: 10.
            fp32(bool)          : If set to false, use half precision during inference (only available for cuda device).
            outscale(int)       : rezising scale to be used (HAVE IN MIND HIS IMPLEMENTATION IS STRICTLI FOR X4)
            device(str)         : Define the device to use
            

        Returns:
            np.ndarray: upscaled image
        """
        print('Upscaling...')
        upscaler_service.get_model(MODEL_PATH, tile, tile_pad, pre_pad, fp32, device)
        img  = upscaler_service.preprocess(np_image)
        try:
            output, _ = upscaler_service.model.enhance(img, outscale=outscale)
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
    #Initialize the model
    UpscalingService.get_model(MODEL_PATH, **{k:v for k,v in KWARGS.items() if k!='outscale'})
    health =  UpscalingService.model is not None
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
    upscaled_img = UpscalingService.predict(np_image, **KWARGS)
    _, img_encoded = cv2.imencode(".png", upscaled_img)
    return StreamingResponse(io.BytesIO(img_encoded), media_type="image/png")
