# Cloud Wine
A scalable wine recommendation engine.

## Motivation for this project:

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

## Build Model
- Local build
```
python3 train/train.py -f './train/config.yaml'
```
- Docker build
```
docker build -t cloudwine-streamlit:v1 -f Dockerfile.app .
```

## Run Streamlit App Locally
```
streamlit run app.py
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
