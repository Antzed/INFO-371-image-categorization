# Use an official TensorFlow runtime as a parent image, with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir numpy pandas matplotlib Pillow scipy scikit-learn

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 6006 available to the world outside this container
EXPOSE 6006

CMD ["tail", "-f", "/dev/null"]
