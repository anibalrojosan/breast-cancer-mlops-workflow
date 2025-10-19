from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    radius_mean: float = Field(ge=0)
    texture_mean: float = Field(ge=0)
    perimeter_mean: float = Field(ge=0)
    area_mean: float = Field(ge=0)
    smoothness_mean: float = Field(ge=0)
    compactness_mean: float = Field(ge=0)
    concavity_mean: float = Field(ge=0)
    concave_points_mean: float = Field(ge=0, alias="concave points_mean")
    symmetry_mean: float = Field(ge=0)
    fractal_dimension_mean: float = Field(ge=0)

    radius_se: float = Field(ge=0)
    texture_se: float = Field(ge=0)
    perimeter_se: float = Field(ge=0)
    area_se: float = Field(ge=0)
    smoothness_se: float = Field(ge=0)
    compactness_se: float = Field(ge=0)
    concavity_se: float = Field(ge=0)
    concave_points_se: float = Field(ge=0, alias="concave points_se")
    symmetry_se: float = Field(ge=0)
    fractal_dimension_se: float = Field(ge=0)

    radius_worst: float = Field(ge=0)
    texture_worst: float = Field(ge=0)
    perimeter_worst: float = Field(ge=0)
    area_worst: float = Field(ge=0)
    smoothness_worst: float = Field(ge=0)
    compactness_worst: float = Field(ge=0)
    concavity_worst: float = Field(ge=0)
    concave_points_worst: float = Field(ge=0, alias="concave points_worst")
    symmetry_worst: float = Field(ge=0)
    fractal_dimension_worst: float = Field(ge=0)

    model_config = {
        "extra": "forbid",          # reject unknown fields
        "populate_by_name": True,   # allow passing pythonic names, but we’ll output aliases
    }