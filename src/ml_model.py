"""
Machine learning model for medical diagnosis using XGBoost with native NaN handling
and consistent categorical value mapping.
"""
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from .config import CBST_PARAMS, MODEL_ARTIFACT_PATH, NUMERIC_FEATURES
import joblib
import os
import numpy as np

class BooleanToYesNoTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to normalize boolean values to 'yes'/'no' strings."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.applymap(lambda v: 'yes' if v is True else ('no' if v is False else ('MISSING' if pd.isna(v) else str(v))))

    def get_feature_names_out(self, input_features=None):
        # Pass column names through unchanged
        return input_features


class MedicalDiagnosisModel:
    """Machine learning model for medical diagnosis."""

    def __init__(self):
        self.pipeline = None
        self.X_columns = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def train(self, X: pd.DataFrame, y: pd.Series, start_index: int = 200, allowed_feature_names: List[str] | None = None) -> None:
        """
        Train the machine learning model with XGBoost and label encoding.
        """
        # Train only on rows from start_index onward
        if start_index is not None and start_index >= 0:
            X = X.iloc[start_index:]
            y = y.iloc[start_index:]
        # Ensure target labels are strings
        y = y.astype(str)

        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        categorical_cols = X.select_dtypes(exclude='number').columns.tolist()

        # Preprocessing pipelines
        numeric_pipeline = "passthrough"
        categorical_pipeline = Pipeline([
            ('bool2yesno', BooleanToYesNoTransformer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        # Full pipeline with XGBoost classifier
        self.pipeline = Pipeline([
            ('preproc', preprocessor),
            ('xgb', xgb.XGBClassifier(**CBST_PARAMS))
        ])

        # Train model:
        # If a subset of features was provided by the user, set all other columns to NaN
        X_train = X.copy()
        if allowed_feature_names is not None:
            allowed_set = set(allowed_feature_names)
            for col in X_train.columns:
                if col not in allowed_set:
                    X_train[col] = np.nan

        self.pipeline.fit(X_train, y_encoded)
        self.X_columns = X.columns.tolist()

        # Persist the trained pipeline and supporting attributes
        self.save(MODEL_ARTIFACT_PATH)

    def predict(self, features: Dict[str, Any], snake_to_orig: Dict[str, str]) -> Tuple[str, Any]:
        """
        Make a prediction and decode label back to original form.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create DataFrame with correct column order
        df_input = pd.DataFrame([features], columns=self.X_columns)
        intersection5 = list(set(NUMERIC_FEATURES) & set(self.X_columns))
        df_input[intersection5] = df_input[intersection5].apply(pd.to_numeric, errors='coerce')
        
        for col in intersection5:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")  
            df_input[col] = df_input[col].fillna("missing") 
        
        for col in NUMERIC_FEATURES:
            try:
                if col in df_input.columns:
                    df_input[col] = df_input[col].apply(pd.to_numeric, errors='coerce')
            except:
                pass
                
        for col in df_input.columns:
            try:
                val = df_input.loc[0, col]
                if str(val).lower() in ['false', 'no', 'absent']:
                    df_input.loc[0, col] = False
                elif str(val).lower() in ['true', 'yes', 'present']:
                    df_input.loc[0, col] = True
            except:
                pass
        
        pred_num = self.pipeline.predict(df_input)[0]
        pred_prob = self.pipeline.predict_proba(df_input)[0]
        
        # Decode to original label
        if pred_num >= len(self.classes_):
            raise ValueError(f"Unknown predicted class index: {pred_num}")
            
        return self.label_encoder.inverse_transform([pred_num])[0], pred_prob

    def save(self, path: str) -> None:
        """Save pipeline, label encoder classes, and X column order to disk."""
        if self.pipeline is None:
            return
        artifact = {
            "pipeline": self.pipeline,
            "classes_": self.classes_,
            "X_columns": self.X_columns,
        }
        joblib.dump(artifact, path)

    def load(self, path: str) -> bool:
        """Load model from disk if available. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        artifact = joblib.load(path)
        self.pipeline = artifact.get("pipeline")
        self.classes_ = artifact.get("classes_")
        self.X_columns = artifact.get("X_columns")
        # Recreate a LabelEncoder with known classes
        if self.classes_ is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.classes_)
        return self.pipeline is not None

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Get feature importances from the trained XGBoost model.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")

        feature_names = self.pipeline.named_steps['preproc'].get_feature_names_out(self.X_columns)
        model = self.pipeline.named_steps['xgb']
        importances = model.feature_importances_

        return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    def interpret_prediction(self, prediction: str, top_n: int = 5) -> str:
        """
        Interpret the prediction by showing top features.
        """
        if self.pipeline is None:
            return "Error: No trained model found."

        top_features = self.get_feature_importance()[:top_n]
        lines = [f"Model predicts: **{prediction}**", f"Top {top_n} most important features:"]
        for feat, imp in top_features:
            lines.append(f"  • {feat}: {imp:.3f}")
        return "\n".join(lines)
