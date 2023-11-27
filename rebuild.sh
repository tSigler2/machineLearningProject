#!/bin/bash

# Define your image and container names
IMAGE_NAME="waterpump_project"
CONTAINER_NAME="waterpump_container"

# Stop and remove the existing container if it exists
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Rebuild the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Run the Docker container with a mounted volume and port forwarding
echo "Running new Docker container..."
docker run -it --name ${CONTAINER_NAME} \
  -p 8888:8888 \
  -v "$PWD":/usr/src/app \
  ${IMAGE_NAME}
