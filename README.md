
# 🩺 Hybrid LLM-XGBoost Medical Diagnosis Pipeline

An advanced, graph-based workflow combining Large Language Models (LLMs) and Machine Learning (XGBoost) for automated medical diagnosis, specifically tailored for appendicitis prediction from unstructured clinical notes.

---

## 🌟 Overview

This project implements a state-of-the-art hybrid pipeline. It leverages the reasoning capabilities of LLMs to extract structured medical features from raw clinical texts. These features are then validated, processed to compute standard clinical scores (e.g., Alvarado and PAS), and finally fed into a trained XGBoost model to predict the probability of appendicitis. The entire orchestration is managed seamlessly using **LangGraph**.

## ✨ Key Features

* **LLM Feature Extraction:** Utilizes powerful LLMs (OpenAI / Groq) to convert unstructured patient notes into structured data.
* **Automated Clinical Scoring:** Automatically calculates standard medical scores (Alvarado Score, Pediatric Appendicitis Score).
* **Robust ML Prediction:** Uses an XGBoost classifier for high-accuracy diagnosis and probability estimation.
* **Graph-Based Orchestration:** Employs LangGraph to create a deterministic, step-by-step state machine for the diagnosis workflow.
* **Fault-Tolerant:** Includes built-in feature validation, NaN handling, and JSON parsing error management.

---

## 🏗️ Architecture & Workflow

The pipeline operates through a directed acyclic graph (DAG) with the following sequential nodes:

1. `extract_features`: Parses clinical text into a JSON feature dictionary using an LLM.
2. `compute_alvarado_score`: Calculates the Alvarado score based on extracted symptoms.
3. `compute_pas_score`: Calculates the Pediatric Appendicitis Score (PAS).
4. `validate_features`: Cleans, formats, and handles missing values for the ML model.
5. `predict_diagnosis`: Feeds the validated features into XGBoost to get the diagnosis and class probabilities.

---

## 📂 Project Structure

```text
medical_diagnosis-main/
│
├── data/                       # Datasets and CSV files
│   ├── dusseldorf_standard.csv # Standardized clinical dataset
│   └── ...
│
├── src/                        # Core source code
│   ├── workflow.py             # LangGraph orchestration
│   ├── llm_client.py           # LLM connection (OpenAI/Groq)
│   ├── ml_model.py             # XGBoost model and preprocessing
│   ├── feature_validator.py    # Feature validation & scoring logic
│   ├── data_loader.py          # Data ingestion utilities
│   ├── utils.py                # Helper functions (JSON extraction, etc.)
│   └── config.py               # Path configurations
│
├── test_batch.py               # Batch testing script for datasets
└── README.md                   # Project documentation
```
