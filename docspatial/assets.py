# assets

"""Bundled reference data.

Replaces the private asset loader the datatypes module used to call. The data
here is small, public and factual (ISO 3166 country names), so it ships with
the package rather than being fetched or looked up in a private store.
"""

from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load_countries():
    """Country names, lowercased, for address detection.

    Returns a plain list — the private version returned a pandas DataFrame with
    a "Country" column, which was the module's only reason to depend on pandas.
    """
    countries_file = DATA_DIR / "countries.txt"
    with countries_file.open(encoding="utf-8") as fh:
        return [line.strip().lower() for line in fh if line.strip()]
