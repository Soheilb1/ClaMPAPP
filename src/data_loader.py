import pandas as pd
from typing import Dict, List, Tuple
from .config import DATASET_PATH, NUMERIC_FEATURES, BOOL_MAPPING, COLUMNS_TO_DROP
from .utils import to_snake_case, preprocess_dataset, create_column_mapping


class DataLoader:
    """Handles loading and preprocessing of medical datasets."""
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        """
        Initialize DataLoader.
        
        Args:
            dataset_path: Path to the dataset file
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.snake_to_orig = {}
        self.nan_columns_global = []
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load and preprocess the dataset.
        
        Returns:
            Preprocessed DataFrame
        """
        # Load raw dataset
        dt = pd.read_csv(self.dataset_path)
        raw_cols = dt.columns.tolist()
        
        # Create column mapping
        self.snake_to_orig = create_column_mapping(raw_cols)
        
        # Convert column names to snake_case
        dt.columns = [to_snake_case(c) for c in dt.columns]
        
        # Preprocess numeric features
        self.dataset = preprocess_dataset(dt, NUMERIC_FEATURES, BOOL_MAPPING)
        
        return self.dataset
    
    def get_cleaned_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get cleaned dataset for model training.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Drop specified columns
        df = self.dataset.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        df.dropna(subset=["diagnosis"],inplace=True)
        
        # Missing values (NaNs) are left as-is because the downstream 
        # XGBoost classifier handles them natively.
        
        # Split features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        return X, y
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get the column mapping dictionary."""
        return self.snake_to_orig
    
    def update_nan_columns(self, features: Dict[str, any]) -> List[str]:
        """
        Update global nan columns list based on current features.
        
        Args:
            features: Dictionary of features
            
        Returns:
            List of column names with NaN values
        """
        df = pd.DataFrame.from_dict(features, orient="index", columns=["value"])
        self.nan_columns_global = df[df["value"].isna()].index.tolist()
        return self.nan_columns_global 