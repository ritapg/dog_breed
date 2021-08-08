import sys
sys.path.insert(0,r'./')
import numpy as np
from glob import glob
from fastapi.testclient import TestClient
from webservice import app, PORT
import os
import timeit

start = timeit.default_timer()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FEATURE_DIR = r'/doog_breed/'
HOST = f"http://localhost:{PORT}/dog_breed"

client = TestClient(app)
dog_files = np.array(glob("data/dog_images/*/*/*"))

input = dog_files[0]
response = client.post(url=HOST, json=input)
print(response.content)

stop = timeit.default_timer()
print('Time: ', stop - start)
