# tests/schema.py
import pandera as pa
from pandera import Column, DataFrameSchema, Check

heart_schema = DataFrameSchema({
    "Age": Column(float, Check.ge(0) & Check.le(120), nullable=False),
    "Sex": Column(int, Check.isin([0,1])),
    "ExerciseAngina": Column(int, Check.isin([0,1])),
})
