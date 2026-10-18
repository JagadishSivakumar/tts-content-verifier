# Normalisation rules

The same function (`src/tts_verifier/normalize.py`) is applied to the prompt and to every ASR
transcript before alignment. Each rule removes a *formatting* difference so that it is not counted
as a TTS error. Each rule has a unit test in `tests/test_core.py`.

| Rule | Example |
|---|---|
| Lower-case; strip punctuation; keep contractions | "Isn't it?" → `isn't it` |
| Split hyphenated compounds | "twenty-five" → `twenty five` |
| Currency → words + unit (pence/cents for decimals) | "£25" → `twenty five pounds`; "$3.50" → `three dollars fifty cents` |
| Times | "9:05" → `nine oh five`; "14:30" → `fourteen thirty` |
| Ordinals | "21st" → `twenty first` |
| Four-digit numbers 1100–2099 read as years | "1987" → `nineteen eighty seven`; "2003" → `two thousand three` |
| Other numbers; commas ignored; decimals digit-by-digit | "1,250" → `one thousand two hundred fifty`; "3.5" → `three point five` |
| "and" inside numbers is optional | "two hundred and fifty" = `two hundred fifty` |
| Percent and units | "40%" → `forty percent`; "ml" → `millilitres`; "g" → `grams` |
| Mixed letter/digit codes spelled out | "QX7" = "Q X 7" → `q x seven` |
| Abbreviations | "Dr" → `doctor`; "St" → `street`; "No. 5" → `number five` |
| British/American spellings | "kilometers" → `kilometres` |

**Known residual ambiguities** (left as false alarms and visible in error analysis rather than hidden
by extra rules): "14:30" spoken as "two thirty", a year-like number that isn't a year, and codes
read as words. Changing a rule means re-running `05_evaluate.py`; note the change here.
