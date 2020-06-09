# Dockerfile
FROM continuumio/miniconda3
WORKDIR /root

# Copies the trainer code
RUN mkdir /root/data
COPY data /root/data
COPY train_model.py /root/train_model.py
COPY utils.py /root/utils.py
COPY models.py /root/models.py
COPY config.yaml /root/config.yaml
COPY requirements.txt /root/requirements.txt

RUN conda install python=3.8.3
RUN apt-get update && apt-get install -y vim g++
RUN pip install -r requirements.txt

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "train_model.py"]
