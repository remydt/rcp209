FROM python:3.8-slim-buster

WORKDIR /usr/src/

# Install necessary packages
RUN apt-get update \
  && apt-get install --yes graphviz \
  && pip3 install --no-cache-dir \
  matplotlib \
  numpy \
  pydotplus \
  scikit-learn \
  scipy

# We will use this image in interactive mode
ENTRYPOINT ["/bin/bash"]
