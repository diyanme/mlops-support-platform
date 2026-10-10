from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    contract_type: str
