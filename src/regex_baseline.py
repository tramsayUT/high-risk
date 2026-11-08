import re
import pandas as pd

SUBSTANCE_TERMS: dict[str, str] = {
    "behavior_alcohol": r"\b(etoh|alcohol\w*|drinks?|drinking|drinker|beers?|wine|liquor|vodka|whiskey)\b",
    "behavior_tobacco": r"\b(tob(acco)?|smok\w*|cig\w*|ppd|pack[\s-]?y(ea)?rs?|py|nicotine|chew\w*|snuff)\b",
    "behavior_drug": r"\b(drugs?|il+i?ci?ts?|elicits?|ivdu|ivda|cocaine|heroin|marijuana|mj|thc|cannabis|crack|opioids?|opiates?|meth\w*|amphetamines?|polysubstance|substance|ecstasy|lsd|pcp)\b",
}

PAST_CUES: str = r"\b(quit|quitting|former(ly)?|ex-?\w*|remote|stopped|sober|sobriety|in recovery|previously|prior|used to|abstinent|clean for|years? ago)\b"
NEVER_CUES: str = r"\b(denies|denied|deny|never|no|not|none|negative|non-?smoker|nondrinker|doesn'?t|does not)\b"
SPLIT: str = r"[.;]+"  # sentence punctuation only
STOP_HEADERS: str = r"\b(family history|fam hx|fhx|physical exam)\b"  # social history ends here
MAX_CUE_DIST: int = 60  

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

    # the cue closest to the substance term wins
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

FAMILY: str = r"(wife|husband|spouse|partner|family|daughters?|sons?|children|kids|mother|father|parents?|sisters?|brothers?|siblings?|girlfriend|boyfriend|fiance\w*|grand\w+|niece|nephew|friends?|roommates?|significant other)"

COMMUNITY_PRESENT: str = (
    rf"\b(lives\b[^.]{{0,60}}\bwith\b(?! (no|none)\b)"   # lives [in X / at home] with ...
    rf"|w/ (his |her )?{FAMILY}"
    rf"|has (a |an |\w+ )?(adult |grown |\w+ )?{FAMILY}"   # has two adult sons, has a girlfriend
    rf"|(his|her|their) {FAMILY}"
    rf"|(re)?married|step-?(children|kids|sons?|daughters?)|grandchildren"
    rf"|{FAMILY}[^.]{{0,40}}(supportive|involved|in the area|nearby|visits?)"
    rf"|support(ive)? system|caregiver)\b"
)
COMMUNITY_ABSENT: str = r"\b(widow(ed|er)?|divorced?|separated|lives alone|estranged|no (close )?(family|friends|support)|(wife|husband|spouse|partner) (died|passed|deceased)|(died|passed away|deceased))\b"
EDUCATION: str = r"\b(college|high school|hs grad\w*|graduated?|degree|university|ged|ph\.?d|masters?|bachelors?|education|school|\d+(th|st|nd|rd) grade|studie[sd]|student)\b"

UNEMPLOYED: str = r"\b(unemployed|retired?|retirement|disabilit\w*|disabled|ssi|ssdi|not (currently )?working|out of work|laid off|lost (his|her) job|does not work|doesn'?t work|formerly worked|previously worked|used to work)\b"
EMPLOYED: str = r"\b(works?|working|employed|employment|job|occupation|profession|career)\b"

HOMELESS: str = r"\b(homeless\w*|undomiciled|shelter|on the streets?|no fixed (address|residence)|couch[\s-]?surf\w*)\b"
HOUSED: str = r"\b(lives (with|alone|in|at|on)|living (with|alone|in|at)|resides|home|house|apartment|apt|condo|nursing home|assisted living|group home|nh resident|rehab facility)\b"


def prep_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    flat: str = re.sub(r"\s+", " ", text)
    return re.split(STOP_HEADERS, flat, maxsplit=1, flags=re.IGNORECASE)[0].lower()


def classify_sdoh(text: str) -> dict[str, int]:
    t: str = prep_text(text)
    # README coding: economics 1=employed, 2=unemployed/retired, 0=none; environment 1=housed, 2=homeless, 0=none
    if re.search(UNEMPLOYED, t):
        economics = 2
    elif re.search(EMPLOYED, t):
        economics = 1
    else:
        economics = 0
    if re.search(HOMELESS, t):
        environment = 2
    elif re.search(HOUSED, t):
        environment = 1
    else:
        environment = 0
    return {
        "sdoh_community_present": int(bool(re.search(COMMUNITY_PRESENT, t))),
        "sdoh_community_absent": int(bool(re.search(COMMUNITY_ABSENT, t))),
        "sdoh_education": int(bool(re.search(EDUCATION, t))),
        "sdoh_economics": economics,
        "sdoh_environment": environment,
    }


def predict_sdoh(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(list(texts.apply(classify_sdoh)))


def predict_all(texts: pd.Series) -> pd.DataFrame:
       return pd.concat([predict_sdoh(texts).reset_index(drop=True), predict_substances(texts).reset_index(drop=True)], axis=1)