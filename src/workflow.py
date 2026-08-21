"""
Workflow orchestration for medical diagnosis using LangGraph.
"""
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from .data_loader import DataLoader
from .llm_client import LLMClient
from .feature_validator import FeatureValidator
from .ml_model import MedicalDiagnosisModel
from .config import MODEL_ARTIFACT_PATH


class State(TypedDict):
    """State definition for the workflow."""
    user_input: str
    features: dict
    imputed_features: dict
    validated_features: dict
    predicted_class: str
    response: str
    prob: str


class MedicalDiagnosisWorkflow:
    """Orchestrates the medical diagnosis workflow."""

    def __init__(self):
        """Initialize the workflow components."""
        self.data_loader = DataLoader()
        self.llm_client = LLMClient()
        self.feature_validator = FeatureValidator()
        self.model = MedicalDiagnosisModel()
        self.workflow = None

    def _extract_features(self, state: State) -> State:
        """Extract features from user input."""
        features = self.llm_client.extract_features(state["user_input"])
        return {"features": features}

    def _compute_alvarado_score(self, state: State) -> State:
        
        """Compute Alvarado score from features."""
        features = state.get("features", {})
        updated_features = self.feature_validator.compute_alvarado_score(features)
        return {"features": updated_features}

    def _compute_pas_score(self, state: State) -> State:
        """Compute PAS score from features."""
        features = state.get("features", {})
        updated_features = self.feature_validator.compute_pas_score(features)
        return {"features": updated_features}


    def _validate_features(self, state: State) -> State:
        
        """Validate and correct features."""
        features = state.get("features", {})
        validated_features = self.feature_validator.validate_features(features)

        print("\n--- validated features ---")
        import pandas as pd
        df = pd.DataFrame.from_dict(validated_features, orient="index", columns=["value"])
        print(df)

        return {"validated_features": validated_features}

    def _predict_diagnosis(self, state: State) -> State:
        
        """Make diagnosis prediction."""
        validated_features = state.get("validated_features", {})

        # Load and prepare data
        X, y = self.data_loader.get_cleaned_dataset()
        snake_to_orig = self.data_loader.get_column_mapping()

        # Load cached model if present; otherwise train once (from sample 201 onward) and save
        loaded = self.model.load(MODEL_ARTIFACT_PATH)
        if not loaded:
            # Only train on the features provided by the user; set others to NaN
            allowed_feature_names = list(validated_features.keys())
            self.model.train(X, y, start_index=200, allowed_feature_names=allowed_feature_names)

        # Make prediction
        prediction, prob = self.model.predict(validated_features, snake_to_orig)

        return {
            "predicted_class": prediction,
            "rf_pipeline": self.model.pipeline,
            "X_columns": self.model.X_columns,
            "prob": prob,
        }

    def _format_response(self, state: State) -> State:
        """Format the final response."""
        prediction = state.get("predicted_class")
        probb = state.get("prob")
        response = f"Based on the information provided, the model predicts: **{prediction}**."
        return {
            "response": response,
            "prob": probb,
        }

    def _interpret_prediction(self, state: State) -> State:
        """Interpret the prediction results."""
        prediction = state.get("predicted_class")
        interpretation = self.model.interpret_prediction(prediction)
        probb = state.get("prob")
        return {
            "response": interpretation,
            "prob": probb,
        }

    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("extract_features", self._extract_features)
        workflow.add_node("compute_alvarado_score", self._compute_alvarado_score)
        workflow.add_node("compute_pas_score", self._compute_pas_score)
        workflow.add_node("validate_features", self._validate_features)
        workflow.add_node("predict_diagnosis", self._predict_diagnosis)
        workflow.add_node("interpret_prediction", self._interpret_prediction)
        workflow.add_node("format_response", self._format_response)

        # Define edges
        workflow.add_edge(START, "extract_features")
        workflow.add_edge("extract_features", "compute_alvarado_score")
        workflow.add_edge("compute_alvarado_score", "compute_pas_score")
        workflow.add_edge("compute_pas_score", "validate_features")
        workflow.add_edge("validate_features", "predict_diagnosis")
        workflow.add_edge("predict_diagnosis", "interpret_prediction")
        workflow.add_edge("interpret_prediction", "format_response")
        workflow.add_edge("format_response", END)

        self.workflow = workflow.compile()
        return self.workflow
    def run(self, user_input: str) -> str:
        """
        Run the complete workflow.
        
        Args:
            user_input: Clinical text input
            
        Returns:
            Diagnosis prediction and interpretation
        """
        if self.workflow is None:
            self.build_workflow()
        
        state = self.workflow.invoke({"user_input": user_input})
        return state.get("response", "No response generated.")  , state.get("prob") 