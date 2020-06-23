# 🌤 Cloud Wine 🍷
A scalable wine recommendation application.
![](streamlit-demo.gif)
## Motivation for this project:
With COVID-19 disrupting the wine industry and consumers moving to online sales, it can be an overwhelming experience to select a wine that matches your taste. This project addresses this headache by recommending wine to users based off similarities in taste notes.

## Setup
Clone repository
```
git clone https://github.com/elmidelange/CloudWine
cd ./CloudWine
```

## Requisites
#### Dependencies
- [Anaconda] (https://docs.anaconda.com/anaconda/install/)
- [Streamlit](streamlit.io)

#### Installation
To install the package above, please run:
```shell
conda create --name cloudwine python=3.8
conda activate cloudwine
pip install -r requirements
```

<!-- ## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
``` -->

## Configs
The config.yaml file contains the final mode parameters for input into the training script. See 'Train Model' section below.

<!-- ## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
``` -->

<!-- ## Run Inference
```
# Example

# Step 1
# Step 2
``` -->

## Train Model
```
python3 train/train.py -f './train/config.yaml'
```
Optional: Docker build
```
cd train
docker build -t cloudwine-train:v1 -f Dockerfile.train .
```

## Run Streamlit App
```
streamlit run app.py
```
Optional: Docker build
```
docker build -t cloudwine-streamlit:v1 -f Dockerfile.app .
docker run -p 80:80 cloudwine-streamlit:v1
```

## Deploy to Google Kubernetes Engine (GKE)
Based off the intruction from Google's 'Deploying a containerized web application' (https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app).

Prerequisites:
1) A Google Cloud (GC) Project with billing enabled.
2) The GC SDK installed (https://cloud.google.com/sdk/docs/quickstarts)
3) Install kubernetes
```
gcloud components install kubectl
```

Set up gcloud tool (GC SDK)
```
export PROJECT_ID = gcp-project-name
export ZONE = gcp-compute-zone (e.g. us-westb-1)

gcloud config set project $PROJECT_ID
gcloud config set compute/zone compute-zone

gcloud auth configure-docker
```

Build and push the container image to GC Container Registery:
```
docker build -t gcp.io/$(PROJECT_ID}/cloudwine-streamlit:v1 -f Dockerfile.app .
docker push gcr.io/${PROJECT_ID}/cloudwine-streamlit:v1
```

Create GKE Cluster
```
gcloud container clusters create cloudwine-cluster --machine-type=n1-highmem-2
gcloud compute instances list
```

Deploy app to GKE
```
kubectl create deployment cloudwine-app --image=gcr.io/${PROJECT_ID}/cloudwine-app:v1
kubectl autoscale deployment cloudwine-app --cpu-percent=80 --min=1 --max=5
kubectl get pods
```

Expose app to internet
```
kubectl expose deployment cloudwine-app --name=cloudwine-app-service --type=LoadBalancer --port 80 --target-port 8080
kubectl get service
```
---
**NOTE**

Streamlit uses a default port of 8501 and is changed to 80 in the .streamlit/config.toml. To use Streamlit's default port change the --target-port parameter to 8501.

---

Deleting the deployment
```
kubectl delete service cloudwine-app-service
gcloud container clusters delete cloudwine-cluster
```



## Analysis
Run the streamlit app and see the 'Model Deep Dive' page for data exploration and experiment results.

![](cloudwine/resources/model_evaluation.png?raw=true)
