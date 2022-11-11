from pydantic import BaseModel, Field

class IrisModel(BaseModel):
    SEPAL_LENGHT: float = Field()
    SEPAL_WIDTH: float = Field()
    PETAL_LENGHT: float = Field()
    PETAL_WIDTH: float = Field()

    class Config:
        schema_extra = {
            "example": {
                "SEPAL_LENGHT": 5.1,
                "SEPAL_WIDTH": 3.5,
                "PETAL_LENGHT": 1.4,
                "PETAL_WIDTH": 0.2,
            }
        }
