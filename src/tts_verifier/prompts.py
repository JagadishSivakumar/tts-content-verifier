"""Prompt set: 100 original prompts in 4 categories, organised into template groups.

Each template group yields several prompts that share structure. Dev/test splitting is done at the
GROUP level so near-duplicate prompts never appear on both sides of the split.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# ---- ordinary: plain conversational sentences (control category) -------------------------------
ORDINARY = {
    "ord_weather": ("The forecast for {place} says {weather} until the end of the week.",
                    [{"place": "the coast", "weather": "light rain"},
                     {"place": "the valley", "weather": "clear skies"},
                     {"place": "the north", "weather": "strong winds"},
                     {"place": "the city centre", "weather": "heavy fog"}]),
    "ord_errand": ("Could you pick up {item} on your way home from {place}?",
                   [{"item": "some fresh bread", "place": "work"},
                    {"item": "the parcel", "place": "the station"},
                    {"item": "a bag of oranges", "place": "the gym"},
                    {"item": "my repaired watch", "place": "town"}]),
    "ord_meeting": ("The {team} meeting has moved to {day} because the room was double booked.",
                    [{"team": "design", "day": "Thursday"}, {"team": "finance", "day": "Monday"},
                     {"team": "support", "day": "Friday"}, {"team": "planning", "day": "Wednesday"}]),
    "ord_story": ("She opened the {object} slowly and found {contents} hidden inside.",
                  [{"object": "old wooden box", "contents": "a folded letter"},
                   {"object": "kitchen drawer", "contents": "a silver key"},
                   {"object": "dusty suitcase", "contents": "a stack of photographs"},
                   {"object": "garden shed", "contents": "a sleeping cat"}]),
    "ord_advice": ("If the {device} stops responding, hold the power button for {n} seconds.",
                   [{"device": "router", "n": "ten"}, {"device": "speaker", "n": "five"},
                    {"device": "tablet", "n": "fifteen"}, {"device": "printer", "n": "twenty"}]),
    "ord_travel": ("We missed the last {transport} and had to walk back along the {route}.",
                   [{"transport": "bus", "route": "river path"},
                    {"transport": "train", "route": "old railway line"},
                    {"transport": "ferry", "route": "harbour wall"},
                    {"transport": "tram", "route": "main road"}]),
    "ord_food": ("The soup needed more {a}, so he added {b} and let it simmer.",
                 [{"a": "salt", "b": "a pinch of pepper"}, {"a": "flavour", "b": "fresh thyme"},
                  {"a": "body", "b": "two potatoes"}, {"a": "heat", "b": "a dried chilli"}]),
}

# ---- numbers_names: normalisation-heavy prompts ------------------------------------------------
NUMBERS_NAMES = {
    "num_money": ("Your refund of {amount} will arrive within {days} working days.",
                  [{"amount": "£25", "days": "3"}, {"amount": "£1,250", "days": "5"},
                   {"amount": "$3.50", "days": "10"}, {"amount": "€480", "days": "7"}]),
    "num_time": ("The appointment with Dr {name} is at {time} on the {date} of {month}.",
                 [{"name": "Okonkwo", "time": "9:05", "date": "3rd", "month": "March"},
                  {"name": "Lindqvist", "time": "14:30", "date": "21st", "month": "June"},
                  {"name": "Ramaswamy", "time": "11:45", "date": "12th", "month": "September"},
                  {"name": "Fitzgerald", "time": "16:15", "date": "2nd", "month": "December"}]),
    "num_year": ("The bridge in {town} was completed in {year} and repaired in {year2}.",
                 [{"town": "Aberystwyth", "year": "1864", "year2": "1987"},
                  {"town": "Llandudno", "year": "1905", "year2": "2003"},
                  {"town": "Penzance", "year": "1931", "year2": "2016"},
                  {"town": "Whitby", "year": "1879", "year2": "1999"}]),
    "num_stats": ("Sales rose by {pct}% to {units} units, the highest since {year}.",
                  [{"pct": "12", "units": "48,300", "year": "2011"},
                   {"pct": "3.5", "units": "907", "year": "1998"},
                   {"pct": "40", "units": "1,000,000", "year": "2008"},
                   {"pct": "7", "units": "65,012", "year": "2019"}]),
    "num_address": ("Please send it to {num} {street} St, flat {flat}, postcode {pc}.",
                    [{"num": "14", "street": "Mill", "flat": "3B", "pc": "BS3 4QT"},
                     {"num": "207", "street": "Queen", "flat": "12", "pc": "CF10 1AB"},
                     {"num": "9", "street": "Harbour", "flat": "41", "pc": "SW1A 2AA"},
                     {"num": "88", "street": "Chapel", "flat": "7C", "pc": "M1 5JQ"}]),
    "num_names": ("{a}, {b} and {c} will present the results to the board.",
                  [{"a": "Siobhan", "b": "Nguyen Thi Lan", "c": "Wojciech"},
                   {"a": "Aoife", "b": "Tadhg", "c": "Oluwaseun"},
                   {"a": "Xiaoming", "b": "Bartholomew", "c": "Ngozi"},
                   {"a": "Krzysztof", "b": "Saoirse", "c": "Thandiwe"}]),
    "num_codes": ("Your booking reference is {ref} and your seat is {seat}.",
                  [{"ref": "QX7 49K", "seat": "14C"}, {"ref": "BR2 0LM", "seat": "3A"},
                   {"ref": "ZT9 81P", "seat": "27F"}, {"ref": "KM4 66D", "seat": "9B"}]),
    "num_measure": ("Mix {a} ml of water with {b} g of flour and bake for {c} minutes.",
                    [{"a": "250", "b": "400", "c": "35"}, {"a": "120", "b": "75", "c": "12"},
                     {"a": "600", "b": "1,000", "c": "50"}, {"a": "30", "b": "15", "c": "8"}]),
}

# ---- repetition_prone: repeated words/phrases, lists, tongue-twisters ----------------------------
REPETITION_PRONE = {
    "rep_list": ("Bring {x}, {x}, and more {x}, because we will need a lot of {x}.",
                 [{"x": "water"}, {"x": "rope"}, {"x": "tape"}, {"x": "paper"}, {"x": "chalk"}]),
    "rep_echo": ("No, no, I said the {a} one, not the {b} one, the {a} one.",
                 [{"a": "red", "b": "blue"}, {"a": "left", "b": "right"},
                  {"a": "small", "b": "large"}, {"a": "first", "b": "second"},
                  {"a": "old", "b": "new"}]),
    "rep_twister": ("{t}",
                    [{"t": "Six slim sleds slid slowly down the slippery slope."},
                     {"t": "Fresh fried fish, fish fresh fried, fried fish fresh."},
                     {"t": "A proper copper coffee pot sat upon a copper tray."},
                     {"t": "Red lorry, yellow lorry, red lorry, yellow lorry."},
                     {"t": "Which wristwatches are Swiss wristwatches, and which are not?"}]),
    "rep_count": ("{seq}, and that is where the counting stops.",
                  [{"seq": "One, two, three, four, five, six, seven"},
                   {"seq": "Ten, nine, eight, seven, six, five, four"},
                   {"seq": "Two, four, six, eight, ten, twelve"},
                   {"seq": "Five, ten, fifteen, twenty, twenty five"},
                   {"seq": "Three, three, three, and then four"}]),
}

# ---- long: multi-sentence passages built from themed sentence banks -----------------------------
LONG_BANKS = {
    "long_harbour": [
        "The harbour woke slowly as the first boats returned with the morning catch.",
        "Gulls circled above the market stalls, waiting for anyone careless with their bread.",
        "A man in a yellow coat argued with a trader about the price of mackerel.",
        "Further along the quay, children dangled crab lines over the edge of the wall.",
        "By nine o'clock the tide had turned and the water was creeping back up the slipway.",
        "The café on the corner propped its door open and the smell of coffee drifted outside.",
        "Two fishermen mended a torn net, passing a flask of tea between them.",
    ],
    "long_lab": [
        "The experiment had failed four times, and nobody could agree on the reason.",
        "Priya suspected the sensor, but the calibration logs looked perfectly normal.",
        "On the fifth attempt they swapped the cable and recorded every reading by hand.",
        "The numbers drifted upward after twenty minutes, exactly as they had before.",
        "It was only when the heating clicked on that someone noticed the pattern.",
        "The room temperature, not the equipment, had been changing the results all along.",
        "They moved the rig away from the radiator and the drift disappeared completely.",
    ],
    "long_train": [
        "The train left the station twelve minutes late, which surprised nobody on board.",
        "Outside the window, fields of wet grass gave way to a grey line of hills.",
        "A conductor walked the length of the carriage, checking tickets and apologising.",
        "Somewhere near the middle of the journey, the lights flickered and went out.",
        "For a moment the only sound was the rattle of the wheels beneath the floor.",
        "Then a child laughed, someone switched on a torch, and the tension broke.",
        "When the power returned, the passengers were talking to each other like old friends.",
    ],
    "long_garden": [
        "Every spring she promised herself that this would be the year the garden behaved.",
        "The roses were pruned, the beds were weeded, and the fence was painted green.",
        "By June the slugs had eaten the lettuces and the mint had swallowed the path.",
        "A fox had dug a neat tunnel under the shed and seemed to be living there happily.",
        "She considered fighting back, then sat down with a cup of tea and watched instead.",
        "The fox watched her too, curled in the shade, entirely unconcerned.",
        "In the end she planted more mint and left a bowl of water by the shed.",
    ],
}


def _expand_templates(table: dict, category: str) -> list[dict]:
    out = []
    for group, (tmpl, fills) in table.items():
        for i, fill in enumerate(fills):
            out.append({"id": f"{group}_{i}", "group": group, "category": category,
                        "text": tmpl.format(**fill)})
    return out


def _long_passages(n_per_bank: int = 5, n_sentences: int = 5, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for group, bank in LONG_BANKS.items():
        for i in range(n_per_bank):
            idx = sorted(rng.sample(range(len(bank)), n_sentences))  # keep narrative order
            out.append({"id": f"{group}_{i}", "group": group, "category": "long",
                        "text": " ".join(bank[k] for k in idx)})
    return out


def build_prompts() -> list[dict]:
    prompts = (_expand_templates(ORDINARY, "ordinary")
               + _expand_templates(NUMBERS_NAMES, "numbers_names")
               + _expand_templates(REPETITION_PRONE, "repetition_prone")
               + _long_passages())
    return prompts


def assign_splits(prompts: list[dict], test_fraction: float = 0.4) -> dict:
    """Group-level split, stratified by category, ordered by a stable hash of the group name.

    Whole template groups go to one side, so near-duplicate prompts never straddle dev/test, and
    every category is represented in both splits.
    """
    by_cat: dict = {}
    for p in prompts:
        by_cat.setdefault(p["category"], set()).add(p["group"])
    split = {}
    for cat, groups in by_cat.items():
        ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        n_test = max(1, round(test_fraction * len(ordered)))
        for i, g in enumerate(ordered):
            split[g] = "test" if i < n_test else "dev"
    return split


def write_prompts(path: Path) -> list[dict]:
    prompts = build_prompts()
    split = assign_splits(prompts)
    for p in prompts:
        p["split"] = split[p["group"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    return prompts
