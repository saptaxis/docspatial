# adapters

"""Convert OCR engine output into the docspatial standard word format.

One function per engine, covering the word-extraction path only. Each takes a
parsed response rather than fetching one, so neither needs credentials or a
vendor SDK beyond what parsing requires.

The core library never imports these. Bring your own words, or use an adapter
if you happen to use that engine.

Both were last verified against their APIs in 2024.
"""
