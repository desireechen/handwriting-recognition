#!/bin/bash

sed 's/tensorflow==/tensorflow-cpu==/' ~/handwriting-recognition/project/requirements.txt > api/requirements.txt

docker build -t text_recognizer_api -f api/Dockerfile .
