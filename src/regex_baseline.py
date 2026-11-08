import re
import pandas as pd

SUBSTANCE_TERMS: dict[str, str] = {
    "behavior_alcohol": r"\b(etoh|alcohol\w*|drinks?|drinking|drinker|beers?|wine|liquor|vodka|whiskey)\b",
    "behavior_tobacco": r"\b(tob(acco)?|smok\w*|cig\w*|ppd|pack[\s-]?y(ea)?rs?|py|nicotine|chew\w*|snuff)\b",
    "behavior_drug": r"\b(drugs?|il+i?ci?ts?|elicits?|ivdu|ivda|cocaine|heroin|marijuana|mj|thc|cannabis|crack|opioids?|opiates?|meth\w*|amphetamines?|polysubstance|substance|ecstasy|lsd|pcp)\b",
}

PAST_CUES: str = r"\b(quit|quitting|former(ly)?|ex-?\w*|remote|stopped|sober|sobriety|in recovery|previously|prior|used to|abstinent|clean for|years? ago)\b"
NEVER_CUES: str = r"\b(denies|denied|deny|never|no|not|none|negative|non-?smoker|nondrinker|doesn'?t|does not)\b"
SPLIT: str = r"[.;]+"  # sentence punctuation only; MIMIC notes are hard-wrapped
STOP_HEADERS: str = r"\b(family history|fam hx|fhx|physical exam)\b"  # social history ends here
MAX_CUE_DIST: int = 60  # characters; a cue farther than this from the term is ignored

PRIORITY: list[int] = [1, 2, 3, 4]  # present > past > never > unsure


def nearest_distance(cue_pattern: str, sentence: str, term_spans: list[tuple[int, int]]) -> int:
    best: int = 10**9
    for cue in re.finditer(cue_pattern, sentence):
        for t_start, t_end in term_spans:
            if cue.end() <= t_start:
                dist = t_start - cue.end()
            elif cue.start() >= t_end:
                dist = cue.start() - t_end
            else:
                dist = 0
            best = min(best, dist)
    return best


def classify_sentence(sentence: str, term_pattern: str) -> int:
    s: str = sentence.lower()
    term_spans: list[tuple[int, int]] = [m.span() for m in re.finditer(term_pattern, s, flags=re.IGNORECASE)]
    past_d: int = nearest_distance(PAST_CUES, s, term_spans)
    never_d: int = nearest_distance(NEVER_CUES, s, term_spans)

    # the cue closest to the substance term wins, so cues about other substances in the same sentence are ignored
    if min(past_d, never_d) <= MAX_CUE_DIST:
        return 2 if past_d <= never_d else 3
    if "?" in s:
        return 4
    return 1


def classify_substance(text: str, term_pattern: str) -> int:
    if not isinstance(text, str):
        return 0
    flat: str = re.sub(r"\s+", " ", text)  # join hard-wrapped lines
    flat = re.split(STOP_HEADERS, flat, maxsplit=1, flags=re.IGNORECASE)[0]
    flat = re.sub(r"\b(no known )?drug allerg\w*", " ", flat, flags=re.IGNORECASE)
    hits: list[int] = [
        classify_sentence(sent, term_pattern)
        for sent in re.split(SPLIT, flat)
        if re.search(term_pattern, sent, flags=re.IGNORECASE)
    ]
    if not hits:
        return 0
    return min(hits, key=PRIORITY.index)


def predict_substances(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        label: texts.apply(classify_substance, term_pattern=pattern)
        for label, pattern in SUBSTANCE_TERMS.items()
    })