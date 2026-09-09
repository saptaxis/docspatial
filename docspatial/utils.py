# utils

"""Small helpers shared across modules."""


def round_to_nearest_multiple(value, multiple):
    """Round ``value`` to the nearest multiple of ``multiple``.

    Used to quantize continuous measurements into comparable buckets — text
    heights into line bands when sorting a page into reading order, and word
    rotation angles into a common orientation when matching a phrase.

    A ``multiple`` of 0 is a no-op, so callers can pass a threshold of 0 to
    mean "do not quantize".
    """
    if not multiple:
        return value
    return round(value / multiple) * multiple
