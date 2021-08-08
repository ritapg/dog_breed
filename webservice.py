FEATURE_DIR = r'./dog_breed/'
import sys
sys.path.insert(0, r'./')
import uvicorn
import time
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import Response
from src.features.detectors import *
import torch

# webservice parameters
HOST = '0.0.0.0'
PORT = 5003
webservice_errors = {'SUCESS':200, 'BAD_REQUEST':400, 'INTERNAL ERROR':500,'NO MODEL':501, 'MODEL FAILED':502}

# import dog breed classifier model
model = torch.load('src/models/model_.pt')

# webservice app
app = FastAPI()
@app.post('/dog_breed')
async def dog_breed_check(request: Request):
    start_time = time.time()
    content = await request.json()
    input_ = str(content)

    # import and show image
    img = Image.open(input_)
    plt.imshow(img)
    plt.show()

    x_transf = image_transforms(input_)

    #check if a dog is detected
    dog = dog_detector_inc(x_transf)

    if dog is True:
        #check breed of dog
        breed = predict_breed_transfer(x_transf, model)
        result = 'dog breed detected:' + breed
        return Response(content=result, status_code='SUCESS', media_type='application/json')

    else:
        result = 'error: No dog nor human face detected'
        return Response(content=result, status_code='SUCESS', media_type='application/json')


if __name__ == '__main__':
    try:
        port = PORT

    except IndexError:
        port = PORT

    uvicorn.run('webservice:app', host=HOST, port=port)
