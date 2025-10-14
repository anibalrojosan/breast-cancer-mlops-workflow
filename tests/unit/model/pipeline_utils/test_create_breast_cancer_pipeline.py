from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from src.model.pipeline_utils import create_breast_cancer_pipeline
from src.model.data_preprocessing import drop_unnecessary_columns

def test_create_breast_cancer_pipeline_returns_pipeline():
    """Test that the create_breast_cancer_pipeline function returns a Pipeline object."""
    pipeline = create_breast_cancer_pipeline()
    assert isinstance(pipeline, Pipeline)

def test_create_breast_cancer_pipeline_has_correct_top_level_steps():
    """Test that pipeline has the expected steps (preprocessor and classifier)."""
    pipeline = create_breast_cancer_pipeline()
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'classifier'

def test_create_breast_cancer_pipeline_has_correct_preprocessor_steps():
    """Test that pipeline has the expected 'preprocessor' steps (drop_cols and scaler)."""
    pipeline = create_breast_cancer_pipeline()
    preprocessor = pipeline.named_steps['preprocessor']
    assert isinstance(preprocessor, Pipeline)
    assert len(preprocessor.steps) == 2
    assert preprocessor.steps[0][0] == 'drop_cols'
    assert preprocessor.steps[1][0] == 'scaler'

def test_drop_cols_configuration():
    """Test that the 'drop_cols' step is configured correctly."""
    pipeline = create_breast_cancer_pipeline()
    drop_cols_transformer = pipeline.named_steps['preprocessor'].named_steps['drop_cols']
    assert drop_cols_transformer.func == drop_unnecessary_columns
    assert drop_cols_transformer.validate == False

def test_scaler_configuration():
    """Test that the 'scaler' step is configured correctly (MinMaxScaler)."""
    pipeline = create_breast_cancer_pipeline()
    scaler_transformer = pipeline.named_steps['preprocessor'].named_steps['scaler']
    assert isinstance(scaler_transformer, MinMaxScaler)

def test_classifier_estimator_configuration():
    """Test that the 'classifier' is a RandomForestClassifier with random_state=42."""
    pipeline = create_breast_cancer_pipeline()
    classifier_estimator = pipeline.named_steps['classifier']
    assert isinstance(classifier_estimator, RandomForestClassifier)
    assert classifier_estimator.random_state == 42