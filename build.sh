#!/bin/bash

# Build the Docker image
docker build -t waterpump_project .

# Run the Docker container with a mounted volume and port forwarding
docker run -it --name waterpump_container \
  -p 8888:8888 \
  -v "$PWD":/usr/src/app \
  waterpump_project
