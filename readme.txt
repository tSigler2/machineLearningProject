A Comparative Study of Na ̈ıve Bayes and k-NN Classifiers on the MNIST Dataset 

Overview
This project utilizes Docker to ensure a consistent and reproducible environment for data science work related to water pump analysis. It includes tools like JupyterLab for interactive analysis and visualization.

Prerequisites
- Docker installed on your machine. [Get Docker](https://docs.docker.com/get-docker/).
- Basic understanding of Docker and command-line interfaces.

Setting Up the Environment

Building the Docker Image
To build the Docker image for the first time, run the `build.sh` script. This script creates a Docker image named `waterpump_project` and starts a container named `waterpump_container`.

```bash
./build.sh
```

Rebuilding the Environment
If you make changes to the environment (such as updating environment.yml) and need to rebuild the Docker image, use the rebuild.sh script. This script will stop and remove the existing container, rebuild the image with the latest changes, and start a new container.

```bash
./rebuild.sh
```

Using the Project
Accessing JupyterLab
After running the `build.sh` or `rebuild.sh` script, JupyterLab will be available at `http://localhost:8888`. The command line will provide a specific token for the session. JupyterLab provides an interactive development environment for working with Jupyter notebooks, code, and data.

Restarting the Same Container 
If you just want to restart the container without making any changes to the Docker image (like if you haven't made any changes to your Dockerfile or environment.yml), you can simply start the container again. You can do this with the following Docker command:

```bash
docker start -ai nb_knn_container
```

Project Structure
- `Dockerfile`: Specifies the Docker image configuration.
- `environment.yml`: Lists all the Conda environment dependencies.
- `build.sh`: Script to build and start the Docker container.
- `rebuild.sh`: Script to rebuild and restart the Docker container with changes.

Adding New Dependencies
To add new dependencies to the project:

1. Update the environment.yml file with the required packages.
2. Run the rebuild.sh script to update the Docker environment.

Contributing
Contributions to the project are welcome. Please ensure that any significant changes are accompanied by corresponding updates in the documentation and Docker environment setup.
