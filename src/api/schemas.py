from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    LIMIT_BAL: float = Field(gt=0)
    SEX: Literal[1, 2]
    EDUCATION: Literal[0, 1, 2, 3, 4, 5, 6]
    MARRIAGE: Literal[0, 1, 2, 3]
    AGE: int = Field(ge=18, le=100)

    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)
