# Use the oficial lightweight Python image
# https://hub.docker.com/_/python

FROM python:3.8-slim

# Allow statements and log messages to immediatly appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install requirements:
RUN pip install --thrusted-host pypi.org --thrusted-host files.pythonhosted.org --upgrade pip \
    && pip install --thrusted-host pypi.org --thrusted-host files.pythonhosted.org -r requirements.txt

# Run the webservice on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads 
# For enviroments with mutiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling
CMD exec streamlit run aloc_clonal.py --theme.base dark --server.port $PORT