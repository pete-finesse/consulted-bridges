"""The paper's screen for facts the model knows: hop 1 (-> bridge), hop 2 (-> answer) and the
composed question (-> answer) must all be completed correctly (silent_dissent.entity_experiments.known_check).
The slim CSV has no alias lists, so answers are matched against the dataset's name plus a few
common short forms; this is stricter than the paper's alias matching."""
from __future__ import annotations

import pandas as pd

from silent_dissent.entity_experiments import known_check
from silent_dissent.model import LM

from . import facts as F

SHORT = {"United States of America": ["United States", "USA", "the United States", "the USA", "America"],
         "United Kingdom": ["UK", "the United Kingdom", "England", "Britain", "Great Britain"],
         "People's Republic of China": ["China"], "Republic of Ireland": ["Ireland"],
         "Netherlands": ["the Netherlands"], "Czech Republic": ["the Czech Republic", "Czechia"]}


def known_facts(h, limit: int | None = None, batch_size: int = 16) -> pd.DataFrame:
    facts = F.load()[:limit] if limit else F.load()
    for f in facts:
        f.e3_aliases = [f.e3, *SHORT.get(f.e3, [])]
    lm = LM.__new__(LM)          # the authors' wrapper, without its multiple-choice letter setup
    lm.model, lm.tok, lm.letters, lm.letter_ids = h.model, h.tok, [], []
    rows = known_check(lm, facts, batch_size)
    df = pd.DataFrame(rows)
    df["known"] = df.hop1 & df.hop2 & df.composed
    df[["hop1_text", "hop2_text", "composed_text"]] = pd.DataFrame(df.pop("texts").tolist(), index=df.index)
    return df
