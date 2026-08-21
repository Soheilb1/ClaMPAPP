import pandas as pd
from typing import Dict, Any, List, Iterable
from .config import CLINICAL_RANGES


def is_numeric(val: Any) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


class FeatureValidator:
    """Validates and corrects extracted features based on clinical ranges."""

    FEATURE_ALIASES = {
        "body_temp": "body_temperature",
        "temperature": "body_temperature",
        "temp": "body_temperature",
        "neutrophil_perc": "neutrophil_percentage",
        "neutrophils_percent": "neutrophil_percentage",
        "loss_of_appetite": "appetite_loss",
        "paedriatic_appendicitis_score": "pediatric_appendicitis_score",
    }

    def __init__(self):
        self.ranges = CLINICAL_RANGES

    @staticmethod
    def _is_present(value: Any) -> bool:
        """Treat None/NaN as missing, but allow False/0 as valid values."""
        return value is not None and not pd.isna(value)

    def _canonicalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Map LLM/external aliases to the canonical feature names used in the pipeline."""
        normalized: Dict[str, Any] = {}

        for key, value in features.items():
            canonical_key = self.FEATURE_ALIASES.get(key, key)

            # Keep the first non-missing value if both alias and canonical arrive.
            if canonical_key in normalized and self._is_present(normalized[canonical_key]):
                continue

            normalized[canonical_key] = value

        return normalized

    def _get_first_present(self, features: Dict[str, Any], keys: Iterable[str]) -> Any:
        for key in keys:
            value = features.get(key)
            if self._is_present(value):
                return value
        return None

    def validate_features(
        self,
        features: Dict[str, Any],
        llm_numeric_values: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Validate features using numeric values provided by the LLM.

        Args:
            features: Dictionary of extracted features
            llm_numeric_values: Dictionary of numeric values from LLM

        Returns:
            Dictionary of validated features where out-of-range numeric values are set to NaN
        """
        features = self._canonicalize_features(features)

        if llm_numeric_values is None:
            llm_numeric_values = {}
        else:
            llm_numeric_values = self._canonicalize_features(llm_numeric_values)

        corrected = {}

        for key, raw_val in features.items():
            if key not in self.ranges:
                corrected[key] = raw_val
                continue

            low, high = self.ranges[key]
            num = None

            llm_val = llm_numeric_values.get(key)
            if is_numeric(llm_val):
                num = float(llm_val)
            elif is_numeric(raw_val):
                num = float(raw_val)

            if num is None:
                corrected[key] = None if raw_val is None else raw_val
                continue

            if not (low <= num <= high):
                corrected[key] = None
                continue

            if key in {"age", "us_number", "appendix_wall_layers"}:
                corrected[key] = int(round(num))
            else:
                corrected[key] = float(num)

        return corrected


    def get_nan_columns(self, features: Dict[str, Any]) -> List[str]:
        """
        Get list of columns with NaN values.

        Args:
            features: Dictionary of features

        Returns:
            List of column names with NaN values
        """
        df = pd.DataFrame.from_dict(features, orient="index", columns=["value"])
        return df[df["value"].isna()].index.tolist()

    def compute_alvarado_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Alvarado score only when all required components are present
        and the score is not already provided.
        """
        features = self._canonicalize_features(features)

        if self._is_present(features.get("alvarado_score")):
            return features
        
        rebound_tenderness = self._get_first_present(features, ["rebound_tenderness"])

        if not self._is_present(rebound_tenderness):
            rebound_candidates = [
                self._get_first_present(features, ["contralateral_rebound_tenderness"]),
                self._get_first_present(features, ["ipsilateral_rebound_tenderness"]),
            ]
            rebound_candidates = [
                value for value in rebound_candidates if self._is_present(value)
            ]

            if rebound_candidates:
                rebound_tenderness = any(value is True for value in rebound_candidates)

        required_values = {
            "migratory_pain": self._get_first_present(features, ["migratory_pain"]),
            "appetite_loss": self._get_first_present(features, ["appetite_loss", "loss_of_appetite"]),
            "nausea": self._get_first_present(features, ["nausea"]),
            "lower_abdominal_pain_right": self._get_first_present(features, ["lower_abdominal_pain_right"]),
            "rebound_tenderness": rebound_tenderness,
            "body_temperature": self._get_first_present(features, ["body_temperature", "body_temp"]),
            "wbc_count": self._get_first_present(features, ["wbc_count"]),
            "neutrophil_percentage": self._get_first_present(
                features, ["neutrophil_percentage", "neutrophil_perc"]
            ),
        }

        if not all(self._is_present(value) for value in required_values.values()):
            return features

        score = 0

        if required_values["migratory_pain"] is True:
            score += 1
        if required_values["appetite_loss"] is True:
            score += 1
        if required_values["nausea"] is True:
            score += 1
        if required_values["lower_abdominal_pain_right"] is True:
            score += 2
        if required_values["rebound_tenderness"] is True:
            score += 1
        if float(required_values["body_temperature"]) > 37.3:
            score += 1
        if float(required_values["wbc_count"]) > 10:
            score += 2
        if float(required_values["neutrophil_percentage"]) > 75:
            score += 1

        features["alvarado_score"] = score
        return features

    def compute_pas_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Pediatric Appendicitis Score only when all required components
        are present and the score is not already provided.
        """
        features = self._canonicalize_features(features)

        score_key = "pediatric_appendicitis_score"
        if self._is_present(features.get(score_key)):
            return features

        required_values = {
            "neutrophilia": self._get_first_present(features, ["neutrophilia"]),
            "appetite_loss": self._get_first_present(features, ["appetite_loss", "loss_of_appetite"]),
            "nausea": self._get_first_present(features, ["nausea"]),
            "migratory_pain": self._get_first_present(features, ["migratory_pain"]),
            "ipsilateral_rebound_tenderness": self._get_first_present(
                features, ["ipsilateral_rebound_tenderness"]
            ),
            "body_temperature": self._get_first_present(features, ["body_temperature", "body_temp"]),
            "wbc_count": self._get_first_present(features, ["wbc_count"]),
            "coughing_pain": self._get_first_present(features, ["coughing_pain"]),
            "contralateral_rebound_tenderness": self._get_first_present(
                features, ["contralateral_rebound_tenderness"]
            ),
        }

        if not all(
            self._is_present(value)
            for key, value in required_values.items()
            if key not in {"coughing_pain", "contralateral_rebound_tenderness"}
        ):
            return features

        score = 0

        if required_values["neutrophilia"] is True:
            score += 1
        if required_values["appetite_loss"] is True:
            score += 1
        if required_values["nausea"] is True:
            score += 1
        if required_values["migratory_pain"] is True:
            score += 1
        if required_values["ipsilateral_rebound_tenderness"] is True:
            score += 1
        if float(required_values["body_temperature"]) > 38:
            score += 1
        if float(required_values["wbc_count"]) > 10:
            score += 1
        if (
            required_values["coughing_pain"] is True
            or required_values["contralateral_rebound_tenderness"] is True
        ):
            score += 2

        features[score_key] = score
        return features
