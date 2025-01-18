# Similar flowers app 
![image](https://github.com/user-attachments/assets/e147a9b6-9f50-4cb3-b40c-d5f3f3c8847d)
## Description
This repository contains code for training a machine learning model to find similar flowers and serving it via a Flask web application.  
The application allows users to submit data, and finds similar flower images from cached database.  
Repository was created with Python 3.9, but most likely will work with lower python and packages versions.  

## Features
- Notebook with model training.
- Flask API to try it with any image (predictions by sending POST requests)
- Docker image to simplify installation

You can run app from repo or docker image

## Run from repo

### Installation
To get started, clone this repository to your local machine:

```bash
git clone https://github.com/xkvma/flowers-rec.git
cd flowers-rec
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```

Install the required packages:
```bash
pip install -r requirements.txt
```
### Training
You can try run ```train.ipynb``` and change something to get your own model

You can:
- add new augmentations
- try other models
- add other datasets
- try different metrics (like precision/recall)
- create onnx model with optimizations for faster inference
- and many other
  
### Run flask app
Make sure to navigate to the directory where app.py is located and run:
```bash

python app.py
# By default, the app will run on http://127.0.0.1:5000/.
```

## Run from Docker image
To try it from docker image simply run
```bash
docker run -d -p 5000:5000 twainsanity/flowers:latest
```

## Make Predictions
After flask app initialization use the following curl command or tool like Postman to make request:
```bash
curl -X POST -F file=@test_images/alex.jpg http://localhost:5000/predict
```
Following responce will contain 5 most similar image paths and similarities scores.


### Example
Let's pass rose and Alex and find similar flowers  
<img src="https://github.com/user-attachments/assets/f10afe71-dab8-43f4-94fe-dafda2ec0e32" width="256">
<img src="https://github.com/user-attachments/assets/bb3f3b8d-1a1c-4b3b-bb67-f8ddd9da04ce" width="256">  

```bash
# Example of request and responce
$ curl -X POST -F file=@test_images/rose.jpg http://localhost:5000/predict
{
"rose/8035910225_125beceb98_n.jpg":1.0,
"rose/8742493617_c2a9bf854f_m.jpg":1.0,
"rose/3268459296_a7346c6b2c.jpg":0.99,
"rose/3664842094_5fd60ee26b.jpg":0.99,
"rose/14408977935_a397e796b8_m.jpg":0.99
}

# rose is similar to roses

$ curl -X POST -F file=@test_images/alex.jpg http://localhost:5000/predict
{
"sunflower/4271193206_666ef60aa0_m.jpg":0.39,
"sunflower/6908789145_814d448bb1_n.jpg":0.38,
"sunflower/24459750_eb49f6e4cb_m.jpg":0.38,
"sunflower/15839183375_49bf4f75e8_m.jpg":0.37,
"sunflower/5330608174_b49f7a4c48_m.jpg":0.37
}

# Alex is similar to sunflowers
```






