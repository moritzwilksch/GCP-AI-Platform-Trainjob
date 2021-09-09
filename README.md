# GCP-AI-Platform-Trainjob
Training a model on GCP AI Platform

## Objective & Idea
This is not about modeling itself, but about utilising GCPs AI-Platform to train a Deep Neural Network (keras) on a GPU instance.

![gcp-aip drawio-2](https://user-images.githubusercontent.com/58488209/132772735-0814afcf-13fe-451d-ab96-8d978a35bb6b.png)


## Prerequisites
1) Setup a venv
2) `pip install -r requirements.txt`
3) Store GCP User keys as `keys.json`
4) Check whether code works with `make train-local`
5) Submit train job with `make train-remote`
