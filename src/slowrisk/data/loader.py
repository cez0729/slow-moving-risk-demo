from pathlib import Path

import pandas as pd

from .validation import validate_dataframe


def load_data(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError("The sanitized demo accepts CSV input only")
    return validate_dataframe(pd.read_csv(source))
