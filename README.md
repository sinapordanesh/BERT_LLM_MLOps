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
    ```
   
## Workflow Overview
This section outlines the core activities that transform raw data into actionable insights and operational models, using advanced machine learning and data processing techniques.

### Data Handling
The data handling process is divided into several key areas: ingestion, preprocessing, and exploratory data analysis.

#### Ingestion
Data is ingested from a public CSV file hosted on GitHub. This stage loads the data into a Pandas DataFrame which acts as the basis for all further operations.

```python
import pandas as pd
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
df = pd.read_csv(DATASET_LOC)
print(df.head())
```

#### Preprocessing
The preprocessing steps involve cleaning the data, transforming text data into a format suitable for modeling, and encoding labels. Text cleaning includes lowercasing, removing stopwords, and other common NLP preprocessing methods.

```python
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")

def clean_text(text, stopwords=STOPWORDS):
    text = text.lower()  # lowercasing
    text = re.sub(r"\b(" + r"|".join(stopwords) + r")\b\s*", "", text)  # remove stopwords
    text = re.sub("[^a-z0-9]+", " ", text)  # keep only alphanumeric characters
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)
```

#### Exploratory Data Analysis (EDA)
EDA involves visualizing and understanding the dataset through various angles to spot trends, anomalies, and underlying patterns.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="tag", data=df)
plt.title("Distribution of Tags")
plt.show()

# Generate a word cloud
from wordcloud import WordCloud
text = " ".join(df["clean_text"].dropna())
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

### Model Training and Evaluation
This part of the workflow deals with setting up the training environment, running the training process, and evaluating the model's performance.

#### Training Setup
We utilize PyTorch and the Hugging Face `transformers` library to set up and fine-tune a pre-trained BERT model.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df["tag"].unique()))

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# DataLoader setup
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

#### Evaluation and Metrics
Model evaluation is crucial to understand the effectiveness of the training. Metrics such as accuracy, precision, recall, and F1-score are calculated to measure performance.

```python
from sklearn.metrics import classification_report

# Predictions
predictions = model.predict(validation_data)
report = classification_report(y_true, predictions, target_names=df["tag"].unique())
print(report)
```

### Model Deployment
Model deployment involves setting up a system that can receive input data, process it through the model, and return predictions.

#### Batch Inference
Batch inference is used for processing data in large batches, suitable for less time-sensitive applications.

```python
# Setup a batch processing pipeline
def batch_predict(data):
    processed_data = preprocess(data)
    predictions = model(processed_data)
    return predictions

batch_data = load_batch_data()
results = batch_predict(batch_data)
```

#### Online Inference
For real-time applications, we set up an online inference system using FastAPI, which provides RESTful endpoints for model interaction.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict/")
async def predict(item: Item):
    processed_text = preprocess(item.text)
    prediction = model(processed_text)
    return {"prediction": prediction}

# Run the API server
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

These sections provide a detailed overview of the entire machine learning workflow, from data preparation to deployment, emphasizing practical implementation and scalability.

## Experiment Tracking
Experiment tracking is essential for managing and comparing different model versions and experimental setups. This project utilizes MLflow for this purpose, which integrates seamlessly with our training scripts.

```python
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Start an MLflow experiment
with mlflow.start_run(run_name='BERT_Experiment') as run:
    # Log parameters and metrics
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", model_accuracy)

    # Save model artifacts
    mlflow.pytorch.log_model(model, "model")

print(f"Experiment details: {mlflow.get_run(run.info.run_id)}")
```

## Hyperparameter Tuning
To optimize the model's performance, we employ Ray Tune for hyperparameter tuning, which allows us to explore a vast space efficiently.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def training_function(config):
    # Initialize model with config parameters
    model.init(config)
    for epoch in range(10):
        train_loss = model.train()
        tune.report(loss=train_loss)  # Report metrics to Ray Tune

# Define the search space
config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64])
}

# Setup the scheduler and execute the tuning
scheduler = ASHAScheduler(max_t=100, grace_period=10)
analysis = tune.run(training_function, config=config, scheduler=scheduler, num_samples=10)
best_trial = analysis.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
```

## Custom Model Logic
Implementing robust model logic is crucial for handling diverse input scenarios and maintaining accuracy across all operational environments.

```python
from ray import serve

@serve.deployment(route_prefix="/predict")
class ModelService:
    def __init__(self, model):
        self.model = model

    def __call__(self, request):
        text = request.json()["text"]
        prediction = self.model.predict(text)
        return {"prediction": prediction}

    def fallback(self, text):
        # Custom logic for uncertain predictions
        if self.model.confidence(text) < 0.5:
            return "Uncertain"
        return self.model.predict(text)

serve.deploy(ModelService)
```


## License
This project is released under the MIT License, which provides flexibility for users to modify and redistribute the software.

## Acknowledgments
Thanks ![MadeWithML](https://madewithml.com/) for providing an awsome free tutorial on MLOps practical project guide. 
Special thanks to the developers of MLflow, Ray, and Hugging Face's Transformers library for their powerful tools.

