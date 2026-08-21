"""
LLM client for feature extraction and processing.
"""
from groq import Groq
from openai import OpenAI
from typing import Dict, Any
from .config import GROQ_API_KEY, OPENAI_API_KEY
from .utils import extract_json_from_text, to_snake_case


class LLMClient:
    """Client for interacting with language models."""

    FEATURE_ALIASES = {
        "body_temp": "body_temperature",
        "temperature": "body_temperature",
        "temp": "body_temperature",
        "neutrophil_perc": "neutrophil_percentage",
        "neutrophils_percent": "neutrophil_percentage",
        "loss_of_appetite": "appetite_loss",
        "lower_right_abd_pain": "lower_abdominal_pain_right",
        "paedriatic_appendicitis_score": "pediatric_appendicitis_score",
    }

    def __init__(self, provider: str = "groq"):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ("groq" or "openai")
        """
        self.provider = provider
        if provider == "groq":
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = "llama-3.1-8b-instant"
        elif provider == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = "gpt-4o-mini-2024-07-18"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _is_present(value: Any) -> bool:
        return value is not None

    def _canonicalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM feature aliases to the canonical pipeline names."""
        normalized: Dict[str, Any] = {}

        for key, value in features.items():
            canonical_key = self.FEATURE_ALIASES.get(key, key)

            # Prefer the first non-null value if both alias and canonical exist.
            if canonical_key in normalized and self._is_present(normalized[canonical_key]):
                continue

            normalized[canonical_key] = value

        return normalized

    def extract_features(self, user_input: str) -> Dict[str, Any]:
        """
        Extract features from clinical text using LLM.

        Args:
            user_input: Clinical text input

        Returns:
            Dictionary of extracted features
        """
        prompt = f"""
        You are an assistant that extracts patient features from clinical text.
        Respond strictly with a JSON object (no extra text, comments, or code blocks)
        with the following keys and types:

        Numeric fields (must be number or null):
        Age, BMI, Height, Weight, Length_of_Stay, Alvarado_Score, Paedriatic_Appendicitis_Score, Appendix_Diameter, Body_Temperature, WBC_Count, Neutrophil_Percentage, Segmented_Neutrophils, RBC_Count, Hemoglobin, RDW, Thrombocyte_Count, CRP, US_Number

        For these numeric fields:
        - Return only numeric values without units.
        - If a value is given as a range, return the midpoint.
        - If given as words (e.g., "thirty-eight"), convert to number.
        - For qualitative descriptions like "normal", return the typical clinical normal value (e.g., normal WBC ~7.0).
        - If no information is present, return null.

        Boolean fields (must be either true, false, or null):
        Appendix_on_US, Migratory_Pain, Lower_Right_Abd_Pain, Contralateral_Rebound_Tenderness, Coughing_Pain, Nausea, Loss_of_Appetite, Neutrophilia, Ketones_in_Urine, RBC_in_Urine, WBC_in_Urine, Dysuria, Stool, Peritonitis, Psoas_Sign, Ipsilateral_Rebound_Tenderness, US_Performed, Free_Fluids, Appendix_Wall_Layers, Target_Sign, Appendicolith, Perfusion, Perforation, Surrounding_Tissue_Reaction, Appendicular_Abscess, Pathological_Lymph_Nodes, Bowel_Wall_Thickening, Conglomerate_of_Bowel_Loops, Ileus, Coprostasis, Meteorism, Enteritis, Gynecological_Findings
        For all Boolean fields:
        - Return only one of: true, false, or null.
        - Treat any explicit negative value as false, regardless of capitalization or formatting.
        - Examples of values that must be interpreted as false: "no", "No", "NO", "false", "False", "FALSE", "negative", "Negative", "absent", "not present", "none", "0".
        - Treat explicit positive values as true, regardless of capitalization or formatting.
        - Examples of values that must be interpreted as true: "yes", "Yes", "YES", "true", "True", "TRUE", "positive", "present", "1".
        - If the information is not mentioned, unclear, or cannot be inferred confidently, return null.


        Map the following to true: "yes", "present", "positive"; to false: "no", "absent", "negative"; if implied, infer accordingly.
        Consider all of these features as boolean values. Any value provided for them must always be normalized strictly to either True or False. No other value besides these two is allowed for these columns.

        Categorical fields (return one of the valid categories or null):
        Sex: "male", "female", null
        Abscess_Location: list anatomical sites or null
        Lymph_Nodes_Location: list anatomical sites or null
        Management: "surgical", "conservative", "discharged", "other", null
        Severity: "mild", "moderate", "severe", null
        Diagnosis: "appendicitis", "no appendicitis", "other", null

        Patient_Note: ignore or return null.

        Instructions:
        - Extract values even if vague or qualitative.
        - If uncertain, make the best reasonable guess rather than null.
        - Return only valid values as per above, do NOT output strings like "normal" for numeric fields (convert to numeric).
        - Always respond strictly with JSON, no explanations.
        Do not extract anything beyond the main features, and under no circumstances add comments such as
        // Not calculated from provided text
        // Not mentioned
        // Typical RBC for a female
        // normal clinical value
        Do not include these at all.
        Do not add any comments or extra text under any circumstances. Answers must be one word only, with no additional text or comments, especially not in the form of //
        System instruction:
        When extracting structured fields, NEVER return an empty array [] or a bracketed string like "[]" or "[ ]" as the field value. If a field would be empty list or exactly a bracketed expression (matches regex ^\\s*\\[.*?\\]\\s*$), return JSON null instead. Example: "Abscess_Location": []  ->  "Abscess_Location": null. Also avoid returning the literal string "[hidden]" or "[ ]" — return null instead. If a field contains bracketed segments inside other text, remove those segments; if nothing remains, return null.

        Text: {user_input}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
        )

        raw_response = response.choices[0].message.content.strip()

        # Extract JSON from response.
        features = extract_json_from_text(raw_response)

        # Convert keys to snake_case first, then normalize aliases.
        features = {to_snake_case(k): v for k, v in features.items()}
        features = self._canonicalize_features(features)

        return features

    def extract_numeric_value(self, text: str) -> float:
        """
        Extract numeric value from text.

        Args:
            text: Text containing numeric value

        Returns:
            Extracted numeric value or None
        """
        prompt = f"""
Extract only the numeric value from the following text and return it as a number:
"{text}"
If there is no number, return null.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        try:
            return float(response.choices[0].message.content.strip())
        except Exception:
            return None
