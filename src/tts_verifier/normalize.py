"""Text normalisation applied identically to the reference prompt and every ASR hypothesis.

The rules are explicit and deterministic so that a "£25" vs "twenty-five pounds" mismatch is
never counted as a TTS error. Every rule here is a measurement decision: document changes in
docs/NORMALISATION.md and re-run evaluation.
"""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]

_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                      "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

# Spoken forms that ASR systems commonly emit for the same written token.
_ABBREVIATIONS = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miz", "st": "street",
    "etc": "et cetera", "vs": "versus", "approx": "approximately", "no": "number",
}
# Only expand "no" when followed by a digit; handled in _expand_abbreviations.
_UNIT_SYMBOLS = {"%": "percent", "km": "kilometres", "kg": "kilograms", "cm": "centimetres",
                 "mm": "millimetres", "ml": "millilitres", "mph": "miles per hour",
                 "g": "grams"}
# British/American spelling variants collapsed to one form.
_SPELLING = {"kilometers": "kilometres", "centimeters": "centimetres",
             "millimeters": "millimetres", "milliliters": "millilitres", "okay": "ok"}


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            words = number_to_words(head) + " " + name
            if rest:
                words += (" and " if rest < 100 else " ") + number_to_words(rest)
            return words
    raise ValueError(n)


def year_to_words(n: int) -> str:
    """1984 -> nineteen eighty four; 2005 -> two thousand and five; 1900 -> nineteen hundred."""
    if 2000 <= n <= 2009:
        return number_to_words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return number_to_words(hi) + " hundred"
    if lo < 10:
        return number_to_words(hi) + " oh " + number_to_words(lo)
    return number_to_words(hi) + " " + number_to_words(lo)


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _expand_number_token(tok: str, ctx_year: bool) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        whole, frac = tok.split(".", 1)
        return number_to_words(int(whole or 0)) + " point " + _digits(frac)
    n = int(tok)
    if ctx_year and 1100 <= n <= 2099:
        return year_to_words(n)
    return number_to_words(n)


def _expand_numbers(text: str) -> str:
    # Currency: £25, $3.50, €10 -> words + unit (pence/cents handled for two-decimal amounts).
    def money(m: re.Match) -> str:
        sym, amount = m.group(1), m.group(2).replace(",", "")
        unit = {"£": ("pound", "pence"), "$": ("dollar", "cents"), "€": ("euro", "cents")}[sym]
        if "." in amount:
            whole, frac = amount.split(".")
            words = number_to_words(int(whole)) + " " + unit[0] + ("" if int(whole) == 1 else "s")
            if int(frac):
                words += " " + number_to_words(int(frac)) + " " + unit[1]
            return " " + words + " "
        n = int(amount)
        return " " + number_to_words(n) + " " + unit[0] + ("" if n == 1 else "s") + " "

    text = re.sub(r"([£$€])(\d[\d,]*(?:\.\d{2})?)", money, text)
    # Times: 3:45 -> three forty five; 9:05 -> nine oh five; 10:00 -> ten
    def clock(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f" {number_to_words(h)} "
        return f" {number_to_words(h)} {'oh ' if mi < 10 else ''}{number_to_words(mi)} "

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", clock, text)
    # Ordinals: 1st, 22nd, 3rd, 15th
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: " " + ordinal_to_words(int(m.group(1))) + " ", text)
    # Percent and units attached or spaced: 40%, 12 km
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    # Years when preceded by "in"/"since"/"from"/"by" or standalone 4-digit 1100-2099.
    def num(m: re.Match) -> str:
        tok = m.group(0)
        is_year = re.fullmatch(r"\d{4}", tok) is not None
        return " " + _expand_number_token(tok, ctx_year=is_year) + " "

    text = re.sub(r"\d[\d,]*(?:\.\d+)?", num, text)
    return text


def _expand_abbreviations(words: list[str]) -> list[str]:
    out: list[str] = []
    for i, w in enumerate(words):
        if w == "no":
            nxt = words[i + 1] if i + 1 < len(words) else ""
            out.append("number" if nxt and nxt[0].isdigit() else w)
        elif w in _ABBREVIATIONS:
            out.extend(_ABBREVIATIONS[w].split())
        elif w in _UNIT_SYMBOLS:
            out.extend(_UNIT_SYMBOLS[w].split())
        else:
            out.append(_SPELLING.get(w, w))
    return out


_NUMBER_WORDS = set(_ONES) | set(_TENS[2:]) | {"hundred", "thousand", "million", "billion"}


def _split_codes(t: str) -> str:
    """Mixed letter/digit codes (QX7, BS3, 14C) -> letters spelled out, digit runs kept.

    ASR systems write the same spoken code as "QX7", "Q X 7" or "QX 7"; splitting makes all of
    them identical after normalisation.
    """
    def split(m: re.Match) -> str:
        if re.fullmatch(r"\d+(st|nd|rd|th)", m.group(0), flags=re.IGNORECASE):
            return m.group(0)                      # ordinals are not codes
        parts = re.findall(r"[A-Za-z]+|\d+", m.group(0))
        out = []
        for p in parts:
            out.extend(list(p) if p.isalpha() else [p])
        return " " + " ".join(out) + " "

    return re.sub(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b", split, t)


def _drop_number_and(words: list[str]) -> list[str]:
    """'two hundred and fifty' == 'two hundred fifty': drop 'and' between number words."""
    out = []
    for i, w in enumerate(words):
        if (w == "and" and 0 < i < len(words) - 1
                and words[i - 1] in {"hundred", "thousand", "million", "billion"}
                and words[i + 1] in _NUMBER_WORDS):
            continue
        out.append(w)
    return out


def normalize(text: str) -> list[str]:
    """Return the normalised word sequence used for alignment."""
    t = text.replace("’", "'").replace("–", " ").replace("—", " ")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)          # hyphenated compounds split
    t = _split_codes(t)
    t = re.sub(r"\bNo\.?\s+(?=\d)", "number ", t, flags=re.IGNORECASE)   # "No. 5" -> "number 5"
    t = _expand_numbers(t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"'(?![a-z])|(?<![a-z])'", " ", t)    # strip quote marks, keep contractions
    words = t.split()
    words = _expand_abbreviations(words)
    words = _drop_number_and(words)
    return [w for w in words if w]
