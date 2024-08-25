# Use an image with CUDA drivers installed (for example, nvidia/cuda)
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04


# Set the working directory inside the container
WORKDIR /app

# Copy the entire current directory to /app inside the container
COPY . /app

# Install Python and venv
RUN apt-get update && apt-get install -y python3.10 python3-pip python3-venv

# Create a virtual environment
RUN python3.10 --version
RUN python3.10 -m venv /app/venv


# Activate the virtual environment and install dependencies
RUN /app/venv/bin/pip install setuptools
RUN /app/venv/bin/pip install --no-cache-dir .

# Set the default Python to be the one in the venv
ENV PATH="/app/venv/bin:$PATH"

# Specify the command to run when starting the container
CMD ["python3", "train.py"]

