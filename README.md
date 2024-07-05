# BERT for Natural Language Processing - MLOps Project

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Workflow Overview](#workflow-overview)
    - [Data Handling](#data-handling)
        - [Ingestion](#ingestion)
        - [Preprocessing](#preprocessing)
        - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Model Training and Evaluation](#model-training-and-evaluation)
        - [Training Setup](#training-setup)
        - [Evaluation and Metrics](#evaluation-and-metrics)
    - [Model Deployment](#model-deployment)
        - [Batch Inference](#batch-inference)
        - [Online Inference](#online-inference)
- [Experiment Tracking](#experiment-tracking)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Custom Model Logic](#custom-model-logic)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This project applies modern MLOps practices to build, deploy, and maintain an NLP model using the BERT architecture. The goal is to efficiently handle natural language processing tasks by leveraging a pre-trained BERT model and enhancing it through fine-tuning, distributed computing, and advanced deployment techniques. The project not only demonstrates a robust machine learning pipeline but also integrates extensive data handling, model evaluation, and deployment strategies ensuring scalability and reproducibility.

## Project Structure
The repository is structured around a main Jupyter notebook which documents and executes the entire MLOps pipeline:
- `start.ipynb`: This notebook contains the comprehensive workflow, including data ingestion, preprocessing, exploratory data analysis, model training, evaluation, and deployment. It serves as the central document where all processes are scripted and annotated for clarity and reproducibility.

## Technologies Used
- **Ray**:  Utilized for its powerful distributed computing capabilities, Ray is integral for handling large datasets and parallelizing data processing and model training.
- **BERT and Transformers**: Using Hugging Face's transformers library, the project harnesses a pre-trained BERT model for NLP tasks, benefiting from the rich, contextual representations that BERT provides.
- **Pandas**:  Employed for data ingestion and manipulation, Pandas is used to handle data frames and perform data cleaning and slicing operations efficiently.
- **Scikit-Learn**: This library is used for splitting the dataset into training and validation sets, ensuring that the model is evaluated on unseen data.
- **MLflow**: For tracking experiments, managing models, and logging all aspects of the machine learning lifecycle, MLflow is used to ensure that each experiment can be reproduced and compared accurately.
- **FastAPI and Ray Serve**: These technologies are used to set up online serving capabilities, allowing the model to process real-time requests through REST API.
- **JAX, Haiku, and PyTorch**: These libraries provide robust, flexible, and efficient computational frameworks for operations on tensors, implementing neural networks, and training them with automatic differentiation.
- **Snorkel**: Used for programmatically building and managing training datasets without manual labeling.

## Setup and Installation
The setup process is designed to be straightforward, involving cloning the repository, setting up the environment, and initializing necessary services:
1. **Clone the Repository:**
   ```bash
    git clone https://github.com/sinapordanesh/BERT_LLM_MLOps.git
    cd BERT_LLM_MLOps
   ```
2. **Environment Setup:**
    ```bash
        pip install -r requirements.txt
    ```
3. **Initialize Ray:**
Ray must be initialized to manage distributed data processing and model training effectively. The initialization code checks if Ray is already running and starts it if not.
    ```bash
        import ray
        if not ray.is_initialized():
            ray.init()
   