# Article titles classification

This is just a simple text classification repo: entering a title and the models output one of the classes "science and technology", "entertainment", "business", "medical". I wrote the models in `pytorch` and `pytorch-lightning`.

I create this repo just to learn using FastAPI and Docker.

Hope you can learn something here. I am quite a novice so any issue is welcomed.

## Installation
Create a virtual environment and run:

```
pip install -r requirements.txt
```

## Run the app
After installing the required packages, run this command from project's root:

```
uvicorn main:app
```

then visit **http://127.0.0.1:8000/docs** to see the SwaggerUI. From there you can try out the app by typing a random article title of your own to see the results.

## Run with Docker
Make sure you are at the root of the project and has Docker installed. Run:

```
docker build -t title-classifier-api
```
After the image is built, run:
```
docker run -p 8000:80 title-classifier-api
```
and visit **http://127.0.0.1:8000/docs** for the SwaggerUI.

