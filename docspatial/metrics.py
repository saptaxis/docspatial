# metrics.py

"""Scoring extracted field values against ground truth.

Not spatial reasoning. It is here because the accounting is specific to document
extraction and does not fall out of a general classification metric.

Three states of ground truth
----------------------------

The distinction between them is the whole design:

- ``None`` or ``NaN``: no ground truth exists for this field. The field is
  excluded from every count, including the denominator. Nothing is claimed
  about it either way.
- ``""``: the correct answer is that the field is empty. A real case that can
  be got right or wrong.
- a value: the correct answer is that value.

A prediction of ``None`` is normalised to ``""``, so failing to answer and
answering "nothing" are the same event.

How a field is booked
---------------------

For each field with ground truth, given ``t`` (truth) and ``p`` (prediction):

    t non-empty, p non-empty, matching        tp
    t non-empty, p non-empty, not matching    fp
    t empty,     p non-empty                  fp
    t non-empty, p empty                      fn
    t empty,     p empty                      tn

A wrong value is booked ``fp`` only, never ``fp`` and ``fn`` together. See the
note on recall below for why that choice does not move any reported number.

What is reported, and over what denominator
-------------------------------------------

    precision           tp / num_pred    of the fields it answered, how many were right
    negative_precision  tn / num_neg     of the fields it left empty, how many should have been
    coverage            num_pred / total how often it attempted an answer at all
    accuracy            correct / total  of every field asked for, how many were right
    recall              = accuracy       see below

``correct`` is ``tp + tn``: a field correctly left blank counts as correct.

``negative_precision`` is the number that shows whether values are being
invented for fields that should be empty. Ordinary precision cannot see that
failure, because an invented value in an empty field never enters its numerator
and is indistinguishable from any other false positive.

``coverage`` exists so that abstaining cannot be mistaken for accuracy. A system
that answers a third of the fields and gets them all right has precision 1.0 and
coverage 0.33, and the pair says more than either alone.

Any score whose denominator is zero is ``-1``, not ``0``, so "no positive
predictions were made" cannot be read as "precision was zero".

Why recall is accuracy
----------------------

Deliberate, and not ``tp / (tp + fn)``.

The retrieval unit here is a required field, not a non-empty value. Precision is
taken over ``num_pred``, the fields that got an answer. Recall is taken over
``total``, every field that was asked for. Under that definition a correct blank
is a successful outcome rather than an absence, which is why ``correct`` includes
``tn``.

With the denominator fixed at ``total``, a field cannot be missed. It is answered
rightly or wrongly, so there is no "failed to retrieve" separate from "got it
wrong" for recall to measure, and it collapses onto accuracy.

The textbook formula is not merely different here, it is misleading.
``tp / (tp + fn)`` drops every ``fp`` from the denominator, including a field
that had a real answer and was given a wrong one. On a set where the interesting
failures are wrong values rather than missing ones, it reports a better number
than was earned.

This is also why booking a wrong value as ``fp`` alone is safe. Under the
textbook formula that choice would move the result; under this one it cannot,
because the field is not correct either way. The ``fp`` and ``fn`` split is
diagnostic: it records how a field failed without changing whether it failed.

Matching
--------

``fuzzy_level`` grades what counts as equal:

    0   exact
    1   case, punctuation and space insensitive
    2   substring-fuzzy at ``fuzzy_threshold``

Level 2 looks for the shorter string inside the longer one, in either direction,
so a value extracted with extra trailing words still matches: "Acme Industries"
against "Acme Industries Limited" passes at level 2 and fails at 0 and 1.

Level 2 reaches ``text.find_string``, which needs nltk.
"""
from .text import filter_punctuation, find_string


def text_is_equal(text1, text2, fuzzy_level=0, fuzzy_threshold=0.75):
    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""

    if fuzzy_level == 0:
        match = text1 == text2
    elif fuzzy_level in [1, 2]:
        text1_ = filter_punctuation(
            text1.lower(), include_spaces=True, include_newline=True
        )
        text2_ = filter_punctuation(
            text2.lower(), include_spaces=True, include_newline=True
        )
        if fuzzy_level == 1:
            match = text1_ == text2_
        else:
            match = is_equal_fuzzy(
                text1_, text2_, threshold=fuzzy_threshold
            )
    return match


def is_equal_fuzzy(true, pred, threshold):
    """Returns boolean on fuzzy match above threshold

    Can potentially weight the accuracies downstream using threshold.
    """
    true_ = "" if true is None else true
    pred_ = "" if pred is None else pred

    if true_ == pred_:
        return True
    # fuzzy match done using find_string
    # which works on finding small str inside larger str
    if len(true_) > len(pred_):
        query_str = pred_
        str_ = true_
    else:
        query_str = true_
        str_ = pred_
    match_str, match_score = find_string(
        str_, query_str, threshold=threshold, lower=True
    )
    if match_str is None:
        # print(true, '-', pred, '===\n')
        return False
    else:
        return True


def custom_pr_metrics(true, pred, fuzzy_level=0, fuzzy_threshold=0.75, debug=False):
    """
    fuzzy_level
    0: exact match (lower)
    1: exact match (lower, remove punct, spaces)
    2: fuzzy match (lower, threshold @ 0.75)

    fuzzy_threshold: applicable when fuzzy_level is 2 for threshold match

    true: list of gts - None or NaN implies missing GT. "" is valid neg GT.
    pred: list of preds - None or NaN implies missing extraction, "" is valid -ve extraction.

    """

    if fuzzy_level not in [0, 1, 2]:
        raise Exception(
            f"fuzzy_level {fuzzy_level} is not recognized. Options are [0, 1, 2]."
        )

    if fuzzy_level == 2:
        if not (min(0, 1) <= fuzzy_threshold <= max(0, 1)):
            raise Exception(
                f"fuzzy_threshold is {fuzzy_threshold}. Needs to be betweek [0, 1]"
            )

    if fuzzy_level in [1, 2]:
        true_ = [
            (
                None
                if ii is None or ii != ii
                else filter_punctuation(str(ii).lower(), include_spaces=True)
            )
            for ii in true
        ]
        pred_ = [
            (
                None
                if ii is None or ii != ii
                else filter_punctuation(str(ii).lower(), include_spaces=True)
            )
            for ii in pred
        ]

    correct = 0
    total = 0
    num_pred = 0
    num_neg = 0

    tp, fp, fn, tn = 0, 0, 0, 0

    correct_ = []
    match_ = []
    for i, (t, p) in enumerate(zip(true, pred)):
        # if gt is None / NaN - gt is invalid
        if debug:
            print(
                f"{i:3}, {t if t is not None else '':20}, {p if p is not None else '' :20}"
            )
        if t is None or t != t:
            correct_.append(None)
            match_.append(None)
            continue

        total += 1
        if p is None or p != p:
            p = ""

        if fuzzy_level == 0:
            match_bool = t == p
        elif fuzzy_level == 1:
            match_bool = true_[i] == pred_[i]
        elif fuzzy_level == 2:
            tt, pp = true_[i], pred_[i]
            tt = tt if tt is not None else "xxx"
            pp = pp if pp is not None else "xxx"
            match_bool = is_equal_fuzzy(tt, pp, threshold=fuzzy_threshold)

        if p and t:
            # both pred and gt are non-empty
            if match_bool:
                tp += 1
                match = "tp"
            else:
                fp += 1
                match = "fp"  # is this correct?
            num_pred += 1
        elif p and not t:
            # pred is non-empty, gt is empty, match_bool = False
            fp += 1
            match = "fp"
            num_pred += 1
        elif not p and t:
            # pred is empty, gt is non-empty, match_bool = False
            fn += 1
            match = "fn"
            num_neg += 1
        elif not p and not t:
            # both pred and gt are empty, match_bool = True
            tn += 1
            match = "tn"
            num_neg += 1

        if match_bool:
            correct_.append(True)
            correct += 1
        else:
            correct_.append(False)
        match_.append(match)

    accuracy = correct / total if total != 0 else -1
    precision = sum([x == "tp" for x in match_]) / num_pred if num_pred != 0 else -1
    negative_precision = (
        sum([x == "tn" for x in match_]) / num_neg if num_neg != 0 else -1
    )
    coverage = num_pred / total if total != 0 else -1

    # recall is accuracy here, deliberately. See the module docstring.
    recall = accuracy

    metrics = {
        "total": total,
        "correct": correct,
        "pred": num_pred,
        "pred_correct": sum([x == "tp" for x in match_]),
        "precision": precision,
        "neg": num_neg,
        "neg_correct": sum([x == "tn" for x in match_]),
        "negative_precision": negative_precision,
        "recall": recall,
        "accuracy": accuracy,
        "coverage": coverage,
    }
    return metrics, correct_, match_
