# Use an official Python runtime as a parent image
FROM python:3.11.4-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Install Certbot
RUN apt-get update && apt-get install -y certbot

# Run collectstatic during the build process
RUN python manage.py collectstatic --noinput

# Make port 80 and 443 available to the world outside this container
EXPOSE 80
EXPOSE 443


# Define the command to run on container start
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "60", "sentiment_analysis_project.wsgi:application"]