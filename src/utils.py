"""
Utility functions for data preprocessing and text processing.
"""
import re
import pandas as pd
from typing import Dict, Any


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case format.
    
    Args:
        name: Input string to convert
        
    Returns:
        String in snake_case format
    """
    # Replace non-alphanumeric characters with underscore
    s = re.sub(r"[^\w]", "_", name)
    # Add underscore between lowercase/numbers and uppercase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Convert to lowercase
    s = s.lower()
    # Replace multiple underscores with single underscore
    s = re.sub(r"_+", "_", s)
    # Remove leading/trailing underscores
    s = s.strip("_")
    return s


def preprocess_dataset(dataset: pd.DataFrame, numeric_features: list, bool_mapping: Dict[str, int]) -> pd.DataFrame:
    """
    Preprocess the dataset by converting columns to appropriate types.
    
    Args:
        dataset: Input DataFrame
        numeric_features: List of numeric feature column names
        bool_mapping: Dictionary mapping boolean strings to integers
        
    Returns:
        Preprocessed DataFrame
    """
    # Ensure numeric_features only includes columns present in the dataset
    numeric_available = [col for col in numeric_features if col in dataset.columns]
    
    # Convert columns to numeric, handling boolean strings
    for col in numeric_available:
        dataset[col] = (dataset[col].astype(str)
                        .str.strip()
                        .str.lower()
                        .replace(bool_mapping))
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    return dataset


def create_column_mapping(raw_columns: list) -> Dict[str, str]:
    """
    Create mapping between original column names and snake_case versions.
    
    Args:
        raw_columns: List of original column names
        
    Returns:
        Dictionary mapping snake_case to original column names
    """
    snake_cols = [to_snake_case(c) for c in raw_columns]
    return dict(zip(snake_cols, raw_columns))


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text response.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON dictionary
    """
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            import json
            return json.loads(json_str)
        except json.JSONDecodeError:
            
            return {}
    return {} 