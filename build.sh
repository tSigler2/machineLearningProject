#!/bin/bash

# Build the Docker image
docker build -t nb_knn_project .

# Run the Docker container with a mounted volume and port forwarding
docker run -it --name nb_knn_container \
  -p 8888:8888 \
  -v "$PWD":/usr/src/app \
  nb_knn_project
