gsutil mb -l australia-southeast1 gs://cloudwine
gsutil cp -r ../train/data/raw gs://cloudwine
gsutil mv gs://cloudwine/raw gs://cloudwine/data
