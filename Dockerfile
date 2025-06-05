FROM python:3.12-slim
# Set the working directory
WORKDIR /application
# Copy the requirements file into the container
COPY . /application
# Install the required packages

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

# Expose the port the app runs on
CMD ["python", "application.py"]