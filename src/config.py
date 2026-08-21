"""
Configuration settings for the medical diagnosis system.
"""
import os
from typing import Dict, List, Tuple
from pathlib import Path

# API Configuration
# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
# Dataset Configuration
DATASET_PATH = "data/Regensburg Pediatric Appendicitis.csv"

# Feature Configuration

NUMERIC_FEATURES = [
    "Age",                           # سن بیمار
    "BMI",                           # شاخص توده بدنی
    "Height",                        # قد
    "Weight",                        # وزن
    "Length_of_Stay",                # مدت زمان بستری
    "Alvarado_Score",                # نمره آلارادو
    "Paedriatic_Appendicitis_Score", # نمره آپاندیسیت کودکان
    "Appendix_Diameter",             # قطر آپاندیس
    "Body_Temperature",              # دمای بدن
    "WBC_Count",                     # شمارش گلبول سفید
    "Neutrophil_Percentage",         # درصد نوتروفیل
    "RBC_Count",                     # شمارش گلبول قرمز
    "Hemoglobin",                    # هموگلوبین
    "RDW",                           # عرض توزیع گلبول قرمز
    "Thrombocyte_Count",             # شمارش پلاکت
    "CRP",                           # CRP
    "US_Number" ,
    "age",
    "bmi",
    "height",
    "weight",
    "length_of_stay",
    "alvarado_score",
    "paedriatic_appendicitis_score",
    "appendix_diameter",
    "body_temperature",
    "wbc_count",
    "neutrophil_percentage",
    "rbc_count",
    "hemoglobin",
    "rdw",
    "thrombocyte_count",
    "crp",
    "us_number"# شماره سونوگرافی
]


# Clinical value ranges for validation
CLINICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "bmi": (10.0, 50.0),
    "height": (50.0, 250.0),
    "weight": (2.0, 300.0),
    "length_of_stay": (0, 365),
    "alvarado_score": (0, 10),
    "pediatric_appendicitis_score": (0, 20),
    "appendix_diameter": (0.0, 20.0),
    "body_temperature": (34.0, 42.5),
    "wbc_count": (0.0, 30.0),
    "neutrophil_percentage": (0.0, 100.0),
    "segmented_neutrophils": (0.0, 100.0),
    "rbc_count": (3.0, 6.5),
    "hemoglobin": (6.0, 20.0),
    "rdw": (10.0, 20.0),
    "thrombocyte_count": (0, 1000),
    "crp": (0.0, 500.0),
    "us_number": (0, 5),
    "appendix_wall_layers": (0, 10),
}

# Boolean mapping for data preprocessing
BOOL_MAPPING = {
    'yes': 1, 'y': 1, 'true': 1, '1': 1,
    'no': 0, 'n': 0, 'false': 0, '0': 0
}

# Columns to drop from dataset
COLUMNS_TO_DROP = [
    "Appendix_Wall_Layers", "Target_Sign", "Appendicolith", "Perfusion", "Perforation",
    "Surrounding_Tissue_Reaction", "Appendicular_Abscess", "Abscess_Location",
    "Pathological_Lymph_Nodes", "Lymph_Nodes_Location", "Bowel_Wall_Thickening",
    "Conglomerate_of_Bowel_Loops", "Ileus", "Coprostasis", "Meteorism",
    "Enteritis", "Gynecological_Findings", "Segmented_Neutrophils", "Patient_Note",  "Severity", "Management"
]
COLUMNS_TO_DROP = [string.lower() for string in COLUMNS_TO_DROP]


# Model Configuration


CBST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist"
}

# Artifact paths
ARTIFACTS_DIR = "artifacts"
Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH = str(Path(ARTIFACTS_DIR) / "diagnosis_model.joblib")