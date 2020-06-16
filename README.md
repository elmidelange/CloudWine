# 🌤 Cloud Wine 🍷
A scalable wine recommendation application.

## Motivation for this project:
With COVID-19 disrupting the wine industry and consumers moving to online sales, it can be an overwhelming experience to select a wine that matches your taste. This project addresses this headache by recommending wine to users based off similarities in taste notes. 

## Setup
Clone repository
```
git clone https://github.com/elmidelange/CloudWine
cd ./CloudWine
```

## Requisites
- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.


#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, please run:
```shell
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

<!-- ## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/) -->


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
- Local build
```
python3 train/train.py -f './train/config.yaml'
```
- Docker build
```
cd train
docker build -t cloudwine-train:v1 -f Dockerfile.train .
```

## Run Streamlit App
- Local build
```
streamlit run app.py
```
- Docker build
```
docker build -t cloudwine-streamlit:v1 -f Dockerfile.app .
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
gcloud container clusters create cloudwine-cluster
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

Deleting the deployment
```
kubectl delete service cloudwine-app-service
gcloud container clusters delete cloudwine-cluster
```



## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
