"""Offline radiology-report → finding extractor (training-time only).

Design principles (aligned with competition constraints):
- Reports may generate training / silver labels ONLY.
- Inference must remain vision-only (see assert_vision_only_inference_inputs).
- Prefer silence over confident false negatives when a finding is unmentioned.
- Emit soft probabilities + confidence + discrete state so strategies B–E can
  consume hard / soft / weighted supervision without changing the backbone.

Multilingual cues cover the dominant train languages (EN/ES/FR/TR/DE/PT/IT/HR/EL
stems). This is intentionally rule-based and reproducible — not an LLM call.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from rsna_knee.labels.targets import REPORT_COL, STUDY_ID_COL, TARGET_COLUMNS

# Discrete evidence states for audit / strategy construction.
# P = explicit positive, N = explicit negative/normal, U = uncertain,
# M = unmentioned / no rule fired.
STATE_POSITIVE = "P"
STATE_NEGATIVE = "N"
STATE_UNCERTAIN = "U"
STATE_MISSING = "M"

PRIOR_SOFT = 0.28  # silent / unmentioned prior (not a confident negative)
UNCERTAIN_SOFT = 0.50


def normalize_report(text: str) -> str:
    """Fold case, diacritics, Turkish İ/I, and separator noise."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text)
    t = t.replace("İ", "i").replace("I", "ı").replace("ı", "i")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold()
    t = t.replace("µ", "μ")
    t = re.sub(r"[_\-/\\|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def report_group_id(text: str) -> str:
    """Stable hash of normalized report text for duplicate-report CV grouping."""
    return hashlib.sha1(normalize_report(text).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Lexicon: positive / negative / uncertain / severity cues per finding.
# Anatomy-sensitive findings require anatomy + pathology co-occurrence.
# ---------------------------------------------------------------------------

_NEGATION = (
    r"(?:no|not|without|denies|denied|absent|absence|negative for|"
    r"unremarkable|intact|normal|preserved|within normal limits|"
    r"sin|ningun[ao]?|ningune|no hay|no se observa|no se identific|"
    r"sin signos|sin evidencia|conservad[ao]|limites normales|"
    r"pas de|sans|aucun[e]?|negatif|normale?|intact|"
    r"kein[e]?|ohne|unauffaellig|regelrecht|"
    r"yok|goeruelmedi|gozlenmedi|normal|"
    r"nema|nije|bez|"
    r"χωρις|δεν|φυσιολογ)"
)

_UNCERTAIN = (
    r"(?:possible|possibly|probable|probably|suggests?|suggestive|"
    r"cannot (?:exclude|be excluded)|equivocal|questionable|suspicious|"
    r"may represent|concern for|appears?|"
    r"posible|probablemente|dudoso|sugiere|no se puede excluir|"
    r"possible|evocateur|douteux|suspect|"
    r"muhtemel|supheli|"
    r"πιθανον|υποπτο)"
)

_SEVERITY_HIGH = (
    r"(?:complete|full[- ]?thickness|high[- ]?grade|severe|marked|large|"
    r"displaced|bucket[- ]?handle|grade\s*[ivx3-4]+|"
    r"completa|grave|sever[ao]|grado\s*[ivx3-4]+|"
    r"complete|sever[e]?|grade\s*[ivx3-4]+|"
    r"tam|ileri|"
    r"πληρης|σοβαρη)"
)

_SEVERITY_LOW = (
    r"(?:trace|mild|minimal|small|slight|low[- ]?grade|fraying|sprain|"
    r"leve|fina|minimo|pequeno|escaso|bajo grado|"
    r"leger|faible|minime|petit|"
    r"hafif|kucuk|"
    r"ηπιο|μικρο)"
)


@dataclass(frozen=True)
class FindingLexicon:
    name: str
    # Patterns that indicate the finding anatomy/entity is being discussed.
    anatomy: tuple[str, ...]
    # Pathology / abnormality cues (matched near anatomy when require_near=True).
    positive: tuple[str, ...]
    # Explicit normality / absence phrases (may be global for some findings).
    negative: tuple[str, ...] = ()
    require_near: bool = True
    window: int = 48


LEXICON: tuple[FindingLexicon, ...] = (
    FindingLexicon(
        name="ACL",
        anatomy=(
            r"\bacl\b",
            r"anterior cruciate",
            r"ligamento cruzado anterior",
            r"\blca\b",
            r"ligament croise anterieur",
            r"vorderes kreuzband",
            r"on capraz",
            r"prednji krizni",
            r"πρόσθιου χιαστού",
            r"πρόσθιο χιαστό",
        ),
        positive=(
            r"tear",
            r"torn",
            r"rupture",
            r"ruptur",
            r"rotura",
            r"deficiencia",
            r"deficient",
            r"sprain",
            r"yirtik",
            r"kopma",
            r"ρήξη",
            r"puknuc",
        ),
        negative=(
            r"acl is intact",
            r"acl intact",
            r"acl is preserved",
            r"acl normal",
            r"lca .*limites normales",
            r"ligamentos cruzados .*limites normales",
            r"cruzados y colaterales dentro de limites normales",
        ),
    ),
    FindingLexicon(
        name="MCL",
        anatomy=(
            r"\bmcl\b",
            r"medial collateral",
            r"ligamento colateral medial",
            r"\blcm\b",
            r"ligament collateral medial",
            r"innenband",
            r"medial collateral ligament",
            r"ic yan bag",
            r"medijalni kolateral",
        ),
        positive=(
            r"tear",
            r"torn",
            r"rupture",
            r"sprain",
            r"rotura",
            r"injury",
            r"lesion",
            r"yirtik",
            r"ρήξη",
        ),
        negative=(
            r"mcl is intact",
            r"mcl intact",
            r"lcm .*sin alteraciones",
            r"colaterales .*limites normales",
            r"ligamentos cruzados y colaterales dentro de limites normales",
        ),
    ),
    FindingLexicon(
        name="Medial Meniscus",
        anatomy=(
            r"medial meniscus",
            r"menisco medial",
            r"menisco interno",
            r"menisque interne",
            r"menisque medial",
            r"innenmeniskus",
            r"medial menisk",
            r"ic menisk",
            r"έσω μηνίσκ",
            r"medijalni menisk",
        ),
        positive=(
            r"tear",
            r"torn",
            r"rupture",
            r"rotura",
            r"fissure",
            r"fisura",
            r"degenerat",
            r"yirtik",
            r"ρήξη",
            r"amputacion",
        ),
        negative=(
            r"medial meniscus is not torn",
            r"medial meniscus .*normal",
            r"menisco medial .*sin signos de rotura",
            r"menisco medial de morfologia y senal conservada",
            r"menisque interne .*sans",
        ),
    ),
    FindingLexicon(
        name="Lateral Meniscus",
        anatomy=(
            r"lateral meniscus",
            r"menisco lateral",
            r"menisco externo",
            r"menisque externe",
            r"menisque lateral",
            r"aussenmeniskus",
            r"lateral menisk",
            r"dis menisk",
            r"έξω μηνίσκ",
            r"lateralni menisk",
        ),
        positive=(
            r"tear",
            r"torn",
            r"rupture",
            r"rotura",
            r"fissure",
            r"fisura",
            r"degenerat",
            r"yirtik",
            r"ρήξη",
            r"amputacion",
        ),
        negative=(
            r"lateral meniscus is not torn",
            r"lateral meniscus .*normal",
            r"menisco lateral .*sin signos",
            r"menisque externe .*sans",
            r"menisco externo .*sin",
        ),
    ),
    FindingLexicon(
        name="Medial OA",
        anatomy=(
            r"medial (?:compartment|tibiofemoral|femorotibial)",
            r"compartimento medial",
            r"femorotibial medial",
            r"artrosis .*medial",
            r"chondr(?:osis|al|opathie).*medial",
            r"medial chondr",
            r"ic kompartman",
            r"έσω διαμέρισμα",
        ),
        positive=(
            r"osteoarthritis",
            r"oa\b",
            r"arthrosis",
            r"artrosis",
            r"chondr",
            r"cartilage loss",
            r"full[- ]?thickness cartilage",
            r"grade\s*[ivx2-4]+",
            r"icrs",
            r"gonarthrose",
            r"kireclenme",
        ),
        negative=(
            r"medial .*cartilage .*normal",
            r"compartimentos femorotibiales sin alteraciones",
            r"no focal chondrosis",
        ),
        window=64,
    ),
    FindingLexicon(
        name="Lateral OA",
        anatomy=(
            r"lateral (?:compartment|tibiofemoral|femorotibial)",
            r"compartimento lateral",
            r"femorotibial lateral",
            r"artrosis .*lateral",
            r"chondr(?:osis|al|opathie).*lateral",
            r"lateral chondr",
            r"dis kompartman",
            r"έξω διαμέρισμα",
        ),
        positive=(
            r"osteoarthritis",
            r"oa\b",
            r"arthrosis",
            r"artrosis",
            r"chondr",
            r"cartilage loss",
            r"full[- ]?thickness cartilage",
            r"grade\s*[ivx2-4]+",
            r"icrs",
            r"gonarthrose",
        ),
        negative=(
            r"lateral .*cartilage .*normal",
            r"no focal chondrosis",
            r"compartimentos femorotibiales sin alteraciones",
        ),
        window=64,
    ),
    FindingLexicon(
        name="PF OA",
        anatomy=(
            r"patellofemoral",
            r"femoro[- ]?patellar",
            r"femoro[- ]?patellaire",
            r"femoropatelar",
            r"patellar cartilage",
            r"trochlear cartilage",
            r"pfj",
            r"chondropathia patellae",
            r"επιγονατιδομηρια",
        ),
        positive=(
            r"osteoarthritis",
            r"oa\b",
            r"arthrosis",
            r"artrosis",
            r"chondr",
            r"cartilage loss",
            r"ulcera condral",
            r"grade\s*[ivx2-4]+",
            r"icrs",
        ),
        negative=(
            r"patellofemoral .*normal",
            r"patellar and trochlear cartilage appears",
            r"cartilago .*patel.*sin",
        ),
        window=64,
    ),
    FindingLexicon(
        name="Effusion",
        anatomy=(
            r"effusion",
            r"joint fluid",
            r"derrame",
            r"epanchement",
            r"gelenkerguss",
            r"efuzyon",
            r"izljev",
            r"υγρό",
            r"αρθρικό υγρό",
        ),
        positive=(
            r"effusion",
            r"derrame",
            r"epanchement",
            r"erguss",
            r"efuzyon",
            r"izljev",
            r"υγρό",
            r"fluid",
        ),
        negative=(
            r"no .*effusion",
            r"no joint effusion",
            r"without effusion",
            r"no evidence of .*effusion",
            r"sin derrame",
            r"no hay derrame",
            r"pas d.?epanchement",
            r"kein erguss",
        ),
        require_near=False,
    ),
    FindingLexicon(
        name="Synovitis",
        anatomy=(
            r"synovit",
            r"synovial thickening",
            r"synovial hypertrophy",
            r"sinovitis",
            r"synovite",
            r"synovialitis",
            r"sinovit",
        ),
        positive=(
            r"synovit",
            r"synovial thickening",
            r"synovial hypertrophy",
            r"sinovitis",
            r"synovite",
            r"sinovit",
        ),
        negative=(
            r"no synovitis",
            r"without synovitis",
            r"sin sinovitis",
            r"pas de synovite",
        ),
        require_near=False,
    ),
    FindingLexicon(
        name="Baker's",
        anatomy=(
            r"baker",
            r"popliteal cyst",
            r"cyste poplite",
            r"quiste popliteo",
            r"kiste poplitea",
            r"baker kisti",
            r"poplitealna cista",
            r"βάκερ",
            r"ιγνυακή κύστη",
        ),
        positive=(
            r"baker",
            r"popliteal cyst",
            r"quiste popliteo",
            r"cyste poplite",
            r"baker kisti",
            r"ιγνυακή",
        ),
        negative=(
            r"no baker",
            r"no popliteal cyst",
            r"without popliteal cyst",
            r"sin quiste de baker",
            r"pas de kyste de baker",
        ),
        require_near=False,
    ),
    FindingLexicon(
        name="Contusion",
        anatomy=(
            r"contusion",
            r"bone bruise",
            r"bone marrow edema",
            r"marrow oedema",
            r"marrow edema",
            r"contusion osea",
            r"contusiones oseas",
            r"edema oseo",
            r"oedeme osseux",
            r"knochenmarkodem",
            r"kemik odemi",
            r"kontuzyon",
            r"οστικό οίδημα",
            r"οστεομυελικό οίδημα",
        ),
        positive=(
            r"contusion",
            r"bone bruise",
            r"marrow (?:o)?edema",
            r"edema oseo",
            r"oedeme osseux",
            r"contusiones",
            r"kontuzyon",
            r"οίδημα",
        ),
        negative=(
            r"no .*contusion",
            r"no bone bruise",
            r"no .*marrow (?:o)?edema",
            r"no hay alteraciones de senal .*medula osea",
            r"sin edema oseo",
        ),
        require_near=False,
    ),
    FindingLexicon(
        name="Fracture",
        anatomy=(
            r"fracture",
            r"fractura",
            r"fraktur",
            r"kirik",
            r"κάταγμα",
            r"prijelom",
            r"osteochondral fracture",
            r"impaction fracture",
        ),
        positive=(
            r"fracture",
            r"fractura",
            r"fraktur",
            r"kirik",
            r"κάταγμα",
            r"prijelom",
            r"fractured",
        ),
        negative=(
            r"no fracture",
            r"without fracture",
            r"no evidence of fracture",
            r"pas de fracture",
            r"sin fractura",
            r"keine fraktur",
            r"kirik yok",
        ),
        require_near=False,
    ),
)


@dataclass
class FindingExtraction:
    state: str
    soft: float
    confidence: float
    severity: float  # 0=none/neg, 0.33=low, 0.66=unspecified, 1.0=high
    negated: bool
    uncertain: bool
    evidence: str


def _compile_any(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_NEG_RE = _compile_any([_NEGATION])
_UNC_RE = _compile_any([_UNCERTAIN])
_SEV_HI_RE = _compile_any([_SEVERITY_HIGH])
_SEV_LO_RE = _compile_any([_SEVERITY_LOW])


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?;:])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _sentence_containing(text: str, start: int, end: int) -> str:
    best = text[max(0, start - 40) : min(len(text), end + 40)]
    for sent in _sentences(text):
        idx = text.find(sent)
        if idx < 0:
            continue
        if idx <= start <= idx + len(sent) or idx <= end <= idx + len(sent):
            return sent
    return best


def _near(
    text: str, a_pat: re.Pattern[str], p_pat: re.Pattern[str], window: int
) -> re.Match[str] | None:
    for am in a_pat.finditer(text):
        lo = max(0, am.start() - window)
        hi = min(len(text), am.end() + window)
        if p_pat.search(text[lo:hi]):
            return am
    for pm in p_pat.finditer(text):
        lo = max(0, pm.start() - window)
        hi = min(len(text), pm.end() + window)
        if a_pat.search(text[lo:hi]):
            return pm
    return None


def extract_finding(text_norm: str, lex: FindingLexicon) -> FindingExtraction:
    """Extract one finding from a normalized report string."""
    anat = _compile_any(lex.anatomy)
    pos = _compile_any(lex.positive)
    neg_phrases = _compile_any(lex.negative) if lex.negative else None

    # Explicit multi-word negative phrases first (high precision).
    if neg_phrases is not None:
        mneg = neg_phrases.search(text_norm)
        if mneg:
            return FindingExtraction(
                state=STATE_NEGATIVE,
                soft=0.12,
                confidence=0.62,
                severity=0.0,
                negated=True,
                uncertain=False,
                evidence=mneg.group(0)[:80],
            )

    if lex.require_near:
        hit = _near(text_norm, anat, pos, lex.window)
        if hit is None:
            # Anatomy mentioned with local negation → explicit negative.
            for am in anat.finditer(text_norm):
                ctx = _sentence_containing(text_norm, am.start(), am.end())
                if _NEG_RE.search(ctx) and not pos.search(ctx):
                    return FindingExtraction(
                        state=STATE_NEGATIVE,
                        soft=0.15,
                        confidence=0.55,
                        severity=0.0,
                        negated=True,
                        uncertain=bool(_UNC_RE.search(ctx)),
                        evidence=ctx[:80],
                    )
            return FindingExtraction(
                state=STATE_MISSING,
                soft=PRIOR_SOFT,
                confidence=0.05,
                severity=0.0,
                negated=False,
                uncertain=False,
                evidence="",
            )
        ctx = _sentence_containing(text_norm, hit.start(), hit.end())
    else:
        m = pos.search(text_norm)
        if m is None:
            return FindingExtraction(
                state=STATE_MISSING,
                soft=PRIOR_SOFT,
                confidence=0.05,
                severity=0.0,
                negated=False,
                uncertain=False,
                evidence="",
            )
        ctx = _sentence_containing(text_norm, m.start(), m.end())
        # Negation must appear in the same sentence, preferably before the cue.
        pre = ctx[: max(0, ctx.lower().find(m.group(0)[:20]))] if m.group(0)[:20] in ctx else ctx
        if _NEG_RE.search(ctx):
            # "no fracture" / "sin derrame" style — negation scopes the cue.
            # Avoid cross-sentence bleed by requiring negation in this sentence
            # and not treating a positive noun after an unrelated "sin" elsewhere.
            neg_m = _NEG_RE.search(ctx)
            cue_at = ctx.find(m.group(0)[: min(12, len(m.group(0)))])
            if neg_m is not None and (cue_at < 0 or neg_m.start() < cue_at + 5):
                # If a clear positive assertion verb pattern exists after negation
                # distance large, keep positive — else negative.
                if cue_at - neg_m.start() < 50:
                    return FindingExtraction(
                        state=STATE_NEGATIVE,
                        soft=0.12,
                        confidence=0.60,
                        severity=0.0,
                        negated=True,
                        uncertain=bool(_UNC_RE.search(ctx)),
                        evidence=ctx[:80],
                    )

    uncertain = bool(_UNC_RE.search(ctx))
    if _SEV_HI_RE.search(ctx):
        severity = 1.0
        soft = 0.90
        conf = 0.78
    elif _SEV_LO_RE.search(ctx):
        severity = 0.33
        soft = 0.62
        conf = 0.68
    else:
        severity = 0.66
        soft = 0.82
        conf = 0.72

    if uncertain:
        return FindingExtraction(
            state=STATE_UNCERTAIN,
            soft=min(soft, UNCERTAIN_SOFT + 0.12),
            confidence=min(conf, 0.45),
            severity=severity,
            negated=False,
            uncertain=True,
            evidence=ctx[:80],
        )

    return FindingExtraction(
        state=STATE_POSITIVE,
        soft=soft,
        confidence=conf,
        severity=severity,
        negated=False,
        uncertain=False,
        evidence=ctx[:80],
    )


def extract_report(report: str) -> dict[str, FindingExtraction]:
    text = normalize_report(report)
    return {lex.name: extract_finding(text, lex) for lex in LEXICON}


def extract_reports_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per study with soft/conf/state columns for all targets.

    Column naming:
      soft__ACL, conf__ACL, state__ACL, sev__ACL, ...
      report_group, n_positive, n_negative, n_uncertain, n_missing
    """
    if REPORT_COL not in train_df.columns:
        raise ValueError("train frame must include Report for offline extraction")

    rows: list[dict] = []
    for _, row in train_df.iterrows():
        sid = str(row[STUDY_ID_COL])
        report = row[REPORT_COL]
        extractions = extract_report(report)
        out: dict = {
            STUDY_ID_COL: sid,
            "report_group": report_group_id(str(report)),
            "report_len": len(str(report)),
        }
        n_p = n_n = n_u = n_m = 0
        for name in TARGET_COLUMNS:
            ex = extractions[name]
            out[f"soft__{name}"] = ex.soft
            out[f"conf__{name}"] = ex.confidence
            out[f"state__{name}"] = ex.state
            out[f"sev__{name}"] = ex.severity
            out[f"evidence__{name}"] = ex.evidence
            if ex.state == STATE_POSITIVE:
                n_p += 1
            elif ex.state == STATE_NEGATIVE:
                n_n += 1
            elif ex.state == STATE_UNCERTAIN:
                n_u += 1
            else:
                n_m += 1
        out["n_positive"] = n_p
        out["n_negative"] = n_n
        out["n_uncertain"] = n_u
        out["n_missing"] = n_m
        rows.append(out)
    return pd.DataFrame(rows)


def soft_matrix(extracted: pd.DataFrame) -> np.ndarray:
    cols = [f"soft__{c}" for c in TARGET_COLUMNS]
    return extracted[cols].to_numpy(dtype=float)


def conf_matrix(extracted: pd.DataFrame) -> np.ndarray:
    cols = [f"conf__{c}" for c in TARGET_COLUMNS]
    return extracted[cols].to_numpy(dtype=float)


def state_matrix(extracted: pd.DataFrame) -> np.ndarray:
    cols = [f"state__{c}" for c in TARGET_COLUMNS]
    return extracted[cols].to_numpy(dtype=object)


def extraction_to_dict(ex: FindingExtraction) -> dict:
    return asdict(ex)
