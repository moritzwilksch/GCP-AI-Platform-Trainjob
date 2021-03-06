SHELL := /bin/bash

    
train-local:
	gcloud ai-platform local train \
  --package-path trainer \
  --module-name trainer.task \
  --job-dir local-training-output 

train-remote:
	gcloud ai-platform jobs submit training job3 \
  --package-path trainer \
  --module-name trainer.task \
  --job-dir gs://moritz-bucket/ \
  --scale-tier basic-gpu \
  --region us-central1 \
  --runtime-version 2.5 \
  --python-version 3.7
