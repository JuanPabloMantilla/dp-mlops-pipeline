import pandas as pd
from zenml import step

@step
def ingest_data() -> pd.DataFrame:
    """Ingests data from the specified path."""
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "income"]
    try:
        df = pd.read_csv('data/adult.data', header=None, names=columns, na_values=' ?', skipinitialspace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Error: 'adult.data' not found in the 'data' folder.")