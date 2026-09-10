# adapters

"""Convert OCR engine output into the docspatial standard word format.

Each adapter is optional and pulls its own engine SDK. The core library never
imports these, so docspatial stays engine-agnostic: bring your own words, or
use an adapter if you happen to use that engine.

Both adapters were last verified against their APIs in 2024.
"""
