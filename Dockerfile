# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and any other necessary files (e.g., README, LICENSE)
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install dependencies from the pyproject.toml file
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Install the necessary packages for the FastAPI app
RUN pip install fastapi "uvicorn[standard]" gunicorn

# Copy the entire project code into the container
COPY . /app

# Copy the serve script into the container
COPY serve /usr/local/bin/serve

# Make the serve script executable
RUN chmod +x /usr/local/bin/serve

# Set environment variable to determine the device (cuda or cpu)
ENV env=prod

# Expose the port that the FastAPI app will run on
EXPOSE 8080

# Set the entrypoint for SageMaker to the serve script
ENTRYPOINT ["serve"]
