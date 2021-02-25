FROM tensorflow/tensorflow

WORKDIR /usr/src/

# Install necessary packages
RUN apt-get update \
  && apt-get install --yes graphviz \
  && pip3 install --no-cache-dir \
  Keras \
  matplotlib \
  numpy \
  pydotplus \
  scikit-learn \
  scipy

# We will use this image in interactive mode
ENTRYPOINT ["/bin/bash"]
