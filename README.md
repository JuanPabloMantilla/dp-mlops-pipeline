# End-to-End MLOps Pipeline with Differential Privacy

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![ZenML](https://img.shields.io/badge/ZenML-0.56.0-purple)
![DVC](https://img.shields.io/badge/DVC-3.63.0-teal)
![MLflow](https://img.shields.io/badge/MLflow-2.11+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![CI Pipeline Status](https://github.com/JuanPabloMantilla/dp-mlops-pipeline/actions/workflows/main.yml/badge.svg)](https://github.com/JuanPabloMantilla/dp-mlops-pipeline/actions)

## Table of Contents
- [Project Overview](#-project-overview)
- [Core MLOps Challenge](#-core-mlops-challenge)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contact](#-contact)

## Project Overview

This repository contains a complete, end-to-end MLOps pipeline designed to train, evaluate, and deploy a machine learning model with **Differential Privacy** guarantees. The project automates the entire ML lifecycle, from data versioning to model deployment, using a modern MLOps stack.

The goal is to showcase a professional-grade MLOps workflow, demonstrating the ability to handle complex dependencies, automate processes with CI/CD, and solve real-world model serialization issues for non-standard model types.

## Core MLOps Challenge

The system is built to automate the lifecycle of a classification model that predicts income level based on census data. The core challenge is to build a robust, reproducible, and automated pipeline around a model that uses a custom, non-standard optimizer from the `tensorflow-privacy` library. This introduces significant real-world challenges in:

* **Dependency Management:** Ensuring compatibility between cutting-edge MLOps tools and older, specialized ML libraries.
* **Model Serialization:** Overcoming failures when passing complex, non-serializable model objects between pipeline steps.
* **Headless Authentication:** Securely managing credentials for cloud services (like DVC remotes) in an automated CI/CD environment.

## Tech Stack

| Component                | Technology                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **Pipeline Orchestration** | [ZenML](https://zenml.io/)                                                                                  |
| **Data & Model Versioning**| [DVC](https://dvc.org/)                                                                                     |
| **Experiment Tracking** | [MLflow](https://mlflow.org/)                                                                               |
| **CI/CD Automation** | [GitHub Actions](https://github.com/features/actions)                                                         |
| **Cloud Storage Backend**| S3-Compatible ([Backblaze B2](https://www.backblaze.com/b2/cloud-storage.html)) with `rclone`                 |
| **ML Frameworks** | [TensorFlow](https://www.tensorflow.org/), [TensorFlow Privacy](https://www.tensorflow.org/responsible_ai/privacy), [Scikit-learn](https://scikit-learn.org/) |
| **Deployment** | Local MLflow Model Server                                                                                   |

## Features

* **Modular Pipeline:** The entire workflow is broken down into logical, reusable steps orchestrated by ZenML.
* **Data Versioning:** The dataset is versioned with DVC and stored in a cloud-based S3 remote, keeping the Git repository lightweight.
* **Experiment Tracking:** All pipeline runs, parameters, and metrics are automatically logged to MLflow.
* **Continuous Integration (CI):** A GitHub Actions workflow automatically triggers on every push/pull-request to install dependencies, pull data, and run the full ZenML pipeline, ensuring the project is always in a working state.
* **Continuous Deployment (CD):** The pipeline automatically deploys the model to a local MLflow server if its accuracy meets a predefined threshold.
* **Robust Artifact Handling:** Implements a robust "pass-the-weights" pattern to overcome complex model serialization issues with the custom Keras optimizer.

## Getting Started

### Prerequisites
* Python 3.10 (This exact version is required to ensure compatibility between all libraries in the locked `requirements.txt` file).
* pip (Python package installer)
* [rclone](https://rclone.org/downloads/) installed and available in your PATH.
* A Backblaze B2 (or other S3-compatible) account.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/JuanPabloMantilla/dp-mlops-pipeline.git
    cd dp-mlops-pipeline
    ```
2.  **Create and activate a virtual environment:**
    ```sh
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Configure `rclone` locally:**
    Run `rclone config` and follow the prompts to set up a remote for your S3-compatible storage. This will generate a `rclone.conf` file on your machine.
    
5.  **Configure DVC to use `rclone`:**
    Update your `.dvc/config` file to point to your `rclone` remote. Then, pull the data.
    ```bash
    # Example .dvc/config content:
    # [core]
    #     remote = myremote
    # ['remote "myremote"']
    #     url = rclone://<your-rclone-remote-name>/<your-bucket-name>
    
    dvc pull
    ```

## Usage

1.  **Initialize ZenML and create the stack:**
    ```sh
    # Initialize the ZenML repository
    zenml init

    # Register the MLflow model deployer
    zenml model-deployer register mlflow_deployer --flavor=mlflow
    
    # Create and set the custom stack
    zenml stack register mlflow_stack -o default -a default -d mlflow_deployer
    zenml stack set mlflow_stack
    ```
2.  **Run the full pipeline:**
    ```sh
    python run.py
    ```
3.  **View the results:**
    * **ZenML Dashboard:** To see the pipeline graph and artifacts, run `zenml up`.
    * **MLflow UI:** To see the experiment tracking details, run `mlflow ui`.

## Project Structure
```
dp-mlops-pipeline/
├── .github/workflows/          # GitHub Actions CI/CD workflow
├── .dvc/                       # DVC metadata, including config
├── data/
│   └── adult.data.dvc          # DVC pointer to the raw data
├── notebooks/
│   └── .gitkeep                # Folder for exploratory data analysis (EDA)
├── src/
│   ├── steps/                  # Individual ZenML pipeline steps
│   │   ├── ingest_data.py
│   │   ├── process_data.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── deployment_trigger.py
│   │   └── deploy_model.py
│   └── pipeline.py             # ZenML pipeline definition
├── .gitignore                  # Specifies intentionally untracked files
├── requirements.txt            # Python package dependencies
└── run.py                      # Main entrypoint to execute the ZenML pipeline
```

## Future Improvements

* **Custom Serving Endpoint:** The current deployment step launches a local `mlflow models serve` process. While functional for this project, it fails in a separate process due to the complex serialization of the `DPKerasAdamOptimizer`. A more robust production solution would be to build a custom **FastAPI** server in the `deploy_model` step. This server would load the model weights (`.h5`) and preprocessor (`.pkl`) and define its own prediction endpoint, completely decoupling the serving environment from MLflow's internal model loading.
* **Formal Testing:** Implement a suite of unit and integration tests for the pipeline steps.
* **Cloud Deployment:** Migrate the local ZenML stack and MLflow server to a cloud-based infrastructure (e.g., on AWS, GCP, or Azure) for a truly production-grade setup.

## Contact
Juan Pablo Mantilla Carreño - juanpablomantilla0@gmail.com

Project Link: [https://github.com/JuanPabloMantilla/dp-mlops-pipeline](https://github.com/JuanPabloMantilla/dp-mlops-pipeline)
