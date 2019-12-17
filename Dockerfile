# base-image declaration & credentials
FROM python:3.6
## For PySpark projects use this image below instead:
# FROM neowaylabs/docker-spark:2.3.1-hadoop-2.9.1

# Show Python version on build step
RUN python -V

# Build application
ARG APP_DIR=/app
WORKDIR ${APP_DIR}
ADD requirements.txt .
# must install Cython outside of requirements.txt due to pip bug
RUN pip install Cython 
RUN pip install h5py
RUN pip --disable-pip-version-check install -r requirements.txt
COPY . ${APP_DIR}
RUN pip --disable-pip-version-check install -e .
RUN chmod -R a+w ${APP_DIR}
ENTRYPOINT ["neoway_nlp"]
