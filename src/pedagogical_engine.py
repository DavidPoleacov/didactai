"""Pedagogical decision layer for the Streamlit MVP.

This is intentionally rule-based around the ML services. The concept document
recommends a hybrid model: ML predictions + controlled pedagogical rules, so the
tutor remains safe, explainable, and not a black-box answer machine.
"""
from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from .exercise_bank import load_review_flag_hashes
except ImportError:
    from exercise_bank import load_review_flag_hashes

DIFF_ORDER = ["1 - bază", "2 - mediu", "3 - consolidare", "4 - avansat"]
METACOGNITIVE_QUESTIONS = [
    "Ce știi deja din problemă și ce trebuie aflat?",
    "De ce ai ales această formulă sau metodă?",
    "Care este primul pas obligatoriu pe care nu trebuie să îl sărim?",
    "Cum poți verifica dacă răspunsul tău are sens?",
    "Poți explica aceeași idee într-o problemă din viața reală?",
]


def normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([=/*^])\s*", r"\1", text)
    return text


_INTEGER_PATTERN = r"[+-]?\d+"
_DECIMAL_PATTERN = r"[+-]?(?:\d+[.,]\d*|[.,]\d+)"
_NUMBER_AT_START = re.compile(rf"^(?P<number>{_INTEGER_PATTERN}/{_INTEGER_PATTERN}|{_DECIMAL_PATTERN}|{_INTEGER_PATTERN})")
_FRACTION_PATTERN = re.compile(rf"^{_INTEGER_PATTERN}/{_INTEGER_PATTERN}$")
_DECIMAL_FULL_PATTERN = re.compile(rf"^{_DECIMAL_PATTERN}$")
_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_NUMERIC_LIKE_PATTERN = re.compile(r"^[+\-\d/.,]+$")
_KNOWN_UNITS = {
    "%", "°", "ani", "baieti", "cm", "cm^2", "cm^3", "cm²", "cm³",
    "g", "kg", "km", "l", "lei", "m", "m^2", "m^3", "m²", "m³",
    "minute", "punct", "puncte", "pungi", "zile", "mii lei",
}


def _answer_text(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def _is_malformed_numeric_key(value: str) -> bool:
    candidate = value
    if "=" in candidate:
        if candidate.count("=") != 1:
            return False
        label, candidate = candidate.split("=", maxsplit=1)
        if not _LABEL_PATTERN.fullmatch(label):
            return False
    if _NUMERIC_LIKE_PATTERN.fullmatch(candidate):
        return _parse_rational(candidate) is None
    match = _NUMBER_AT_START.match(candidate)
    return bool(match and _parse_rational(match.group("number")) is None)


def _parse_rational(value: str) -> Fraction | None:
    try:
        if _FRACTION_PATTERN.fullmatch(value):
            numerator, denominator = value.split("/", maxsplit=1)
            denominator_value = int(denominator)
            if denominator_value == 0:
                return None
            return Fraction(int(numerator), denominator_value)
        if _DECIMAL_FULL_PATTERN.fullmatch(value):
            return Fraction(Decimal(value.replace(",", ".")))
        if re.fullmatch(_INTEGER_PATTERN, value):
            return Fraction(int(value), 1)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def _parse_scalar(value: str) -> tuple[str | None, Fraction, str] | None:
    label = None
    scalar = value
    if "=" in scalar:
        if scalar.count("=") != 1:
            return None
        label, scalar = scalar.split("=", maxsplit=1)
        if not _LABEL_PATTERN.fullmatch(label):
            return None

    match = _NUMBER_AT_START.match(scalar)
    if not match:
        return None
    numeric_part = match.group("number")
    rational = _parse_rational(numeric_part)
    if rational is None:
        return None
    unit = scalar[match.end():].strip()
    if unit and unit not in _KNOWN_UNITS:
        return None
    return label, rational, unit


def _looks_like_numeric_key(value: str) -> bool:
    scalar = value
    if "=" in value:
        if value.count("=") != 1:
            return False
        label, scalar = value.split("=", maxsplit=1)
        if not _LABEL_PATTERN.fullmatch(label):
            return False
    return bool(_NUMERIC_LIKE_PATTERN.fullmatch(scalar))


def _has_unsupported_numeric_unit(value: str) -> bool:
    scalar = value
    labeled = "=" in scalar
    if labeled:
        if scalar.count("=") != 1:
            return False
        label, scalar = scalar.split("=", maxsplit=1)
        if not _LABEL_PATTERN.fullmatch(label):
            return False
    match = _NUMBER_AT_START.match(scalar)
    if not match:
        return False
    suffix = scalar[match.end():]
    if not suffix:
        return False
    # A suffix on a labeled scalar or a whitespace-delimited unit is not
    # accepted as exact text when the unit is outside the supported list.
    return labeled or suffix[:1].isspace()


def answer_result_is_verified(result: Dict) -> bool:
    """Return whether an answer result is a definite correct/incorrect verdict."""
    return result.get("status") in {"correct", "incorrect"}


def tutor_answer_can_be_recorded(result: Dict, exercise_completed: bool) -> bool:
    """Guard Tutor state changes against unverified answers and duplicate completion."""
    return answer_result_is_verified(result) and not exercise_completed


def evaluate_answer(student_answer: str, expected_answer: str) -> Dict:
    """Conservatively compare complete answers without evaluating student input.

    Exact normalized text and exact rational scalar equivalence are supported.
    Labels and units must match. General symbolic equivalence, unit conversion,
    sets, intervals, and multi-part answers require review unless text matches exactly.
    """
    student_text = _answer_text(student_answer)
    expected_text = _answer_text(expected_answer)
    student = normalize_answer(student_text or "")
    expected = normalize_answer(expected_text or "")

    status = "needs_review"
    if not student:
        status = "no_answer"
        feedback = "Nu ai introdus încă un răspuns. Începe cu ce se cere și cu datele cunoscute."
    elif expected_text is None or not expected:
        status = "invalid_key"
        feedback = "Cheia de răspuns lipsește sau este goală; acest răspuns nu poate fi evaluat."
    elif len(expected) > 4096:
        status = "invalid_key"
        feedback = "Cheia depășește limita pentru verificarea automată; este necesară verificarea manuală."
    elif _is_malformed_numeric_key(expected):
        status = "invalid_key"
        feedback = "Cheia numerică este malformată și nu poate fi evaluată în siguranță."
    elif len(student) > 4096:
        feedback = "Răspunsul este prea lung pentru verificarea automată; este necesară verificarea manuală."
    else:
        expected_scalar = _parse_scalar(expected)
        student_scalar = _parse_scalar(student)
        if student == expected and expected_scalar is None and _has_unsupported_numeric_unit(expected):
            feedback = "Unitatea sau formatul numeric nu este recunoscut; răspunsul necesită clarificare."
        elif student == expected:
            status = "correct"
            feedback = "Răspunsul se potrivește cu cheia. Acum justifică metoda, nu doar rezultatul."
        elif expected_scalar is not None and student_scalar is not None:
            expected_label, expected_value, expected_unit = expected_scalar
            student_label, student_value, student_unit = student_scalar
            if expected_label != student_label or expected_unit != student_unit:
                feedback = "Verifică eticheta variabilei și unitățile; răspunsul necesită clarificare."
            elif student_value == expected_value:
                status = "correct"
                feedback = "Răspunsul este numeric echivalent cu cheia. Acum justifică metoda."
            else:
                status = "incorrect"
                feedback = "Răspunsul numeric nu coincide. Verifică semnele, operațiile și valoarea finală."
        elif expected_scalar is not None and _looks_like_numeric_key(student):
            feedback = "Răspunsul nu are un format numeric compatibil; este necesară verificarea manuală."
        else:
            feedback = "Răspunsul nu poate fi comparat în siguranță de verificator; este necesară verificarea manuală."

    return {
        "correct": status == "correct",
        "verified": status in {"correct", "incorrect"},
        "status": status,
        "feedback": feedback,
    }


def automatically_gradable_bank(data: pd.DataFrame) -> pd.DataFrame:
    """Keep format-compatible rows except separately reviewed semantic exclusions."""
    compatible = format_compatible_bank(data)
    if "problem_hash" not in compatible.columns:
        return compatible
    excluded_hashes = load_semantic_grading_exclusion_hashes()
    hashes = compatible["problem_hash"].fillna("").astype(str).str.strip()
    return compatible.loc[~hashes.isin(excluded_hashes)].copy()


def format_compatible_bank(data: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose stored key the conservative checker can verify against itself."""
    if "Raspunsul" not in data.columns:
        return data.iloc[0:0].copy()
    eligible = data["Raspunsul"].map(
        lambda answer: evaluate_answer(answer, answer)["status"] == "correct"
    )
    return data.loc[eligible].copy()


def split_solution_steps(solution: str) -> List[str]:
    text = "" if solution is None or pd.isna(solution) else str(solution).strip()
    if not text:
        return []
    # Split on semicolons, line breaks, or sentence boundaries while keeping useful chunks.
    parts = re.split(r"(?:\n+|;|\.\s+)", text)
    steps = [p.strip(" .;\n\t") for p in parts if len(p.strip()) >= 4]
    return steps if steps else [text[:240]]


HINT_DISPLAY_CORRECTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "hint_display_corrections.csv"
)
SEMANTIC_GRADING_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "semantic_grading_exclusions.csv"
)
HINT_REVIEW_MESSAGE = "Indiciul salvat este retras temporar pentru verificare editorială."
_SUSPICIOUS_NEGATIVE_SQUARE = re.compile(
    r"(?:Δ|delta)[^;]*?-\s*\d+\s*(?:²|\^2)", re.IGNORECASE
)
_QUADRATIC_COEFFICIENTS = re.compile(
    r"(?<![\w])(?P<a>[+-]?\s*(?:\d+\s*)?)x(?:²|\^2)"
    r"\s*(?P<b>[+-]\s*(?:\d+\s*)?)x(?:\s*(?P<c>[+-]\s*\d+))?",
    re.IGNORECASE,
)
_SOURCE_DISCRIMINANT = re.compile(
    r"^\s*Δ\s*=\s*(?:b²\s*-\s*4ac\s*=\s*)?"
    r"(?P<b>-?\d+)\s*(?:²|\^2)\s*-\s*4[·*](?P<a>[+-]?\d+)[·*](?P<c>[+-]?\d+)"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_CORRECTED_DISCRIMINANT = re.compile(
    r"^\s*Δ\s*=\s*(?:b²\s*-\s*4ac\s*=\s*)?"
    r"\((?P<b>[+-]\d+)\)\s*(?:²|\^2)\s*-\s*4[·*](?P<a>[+-]?\d+)[·*](?P<c>[+-]?\d+)"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_FINAL_INTEGER = re.compile(r"=\s*(-?\d+)\s*$")


@lru_cache(maxsize=1)
def load_hint_display_corrections() -> Dict[str, Dict[str, str]]:
    """Load the reviewed, source-hash-validated hint display overlay."""
    required = {
        "stable_index", "problem_hash", "step_number", "original_step", "corrected_step",
        "rationale", "source_step_sha256",
    }
    try:
        with HINT_DISPLAY_CORRECTIONS_PATH.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Hint correction overlay is missing columns: {', '.join(sorted(missing))}")
            records = {}
            for row in reader:
                stable_index = row["stable_index"].strip()
                if not stable_index.isdecimal() or stable_index in records:
                    raise ValueError(f"Invalid or duplicate hint correction index: {stable_index!r}")
                if not all(row[column].strip() for column in required - {"corrected_step"}):
                    raise ValueError(f"Incomplete hint correction metadata at index {stable_index}.")
                if not row["corrected_step"].strip():
                    raise ValueError(f"Missing corrected hint step at index {stable_index}.")
                if not row["step_number"].isdecimal() or int(row["step_number"]) < 1:
                    raise ValueError(f"Invalid source step number at stable index {stable_index}.")
                if not re.fullmatch(r"[0-9a-f]{64}", row["source_step_sha256"]):
                    raise ValueError(f"Invalid source-step SHA-256 at stable index {stable_index}.")
                records[stable_index] = row
    except OSError as exc:
        raise ValueError(f"Unable to load reviewed hint corrections: {exc}") from exc
    return records


@lru_cache(maxsize=1)
def load_semantic_grading_exclusion_hashes() -> frozenset[str]:
    """Load separate, reviewed answer-semantics exclusions by stable problem hash."""
    required = {"stable_index", "problem_hash", "reason"}
    try:
        with SEMANTIC_GRADING_EXCLUSIONS_PATH.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Semantic grading exclusions are missing columns: {', '.join(sorted(missing))}"
                )
            records = list(reader)
    except OSError as exc:
        raise ValueError(f"Unable to load semantic grading exclusions: {exc}") from exc
    hashes = set()
    for row in records:
        stable_index = row["stable_index"].strip()
        problem_hash = row["problem_hash"].strip()
        if not stable_index.isdecimal() or not problem_hash or not row["reason"].strip():
            raise ValueError(f"Incomplete semantic grading exclusion at index {stable_index!r}.")
        if problem_hash in hashes:
            raise ValueError(f"Duplicate semantic grading exclusion hash at index {stable_index}.")
        hashes.add(problem_hash)
    return frozenset(hashes)


def _parse_coefficient(value: str | None, default: int) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"\s+", "", value)
    if cleaned in {"", "+"}:
        return default
    if cleaned == "-":
        return -default
    return int(cleaned)


def _quadratic_coefficients(problem: str) -> tuple[int, int, int] | None:
    match = _QUADRATIC_COEFFICIENTS.search(str(problem))
    if not match:
        return None
    try:
        coefficients = (
            _parse_coefficient(match.group("a"), 1),
            _parse_coefficient(match.group("b"), 1),
            _parse_coefficient(match.group("c"), 1),
        )
    except ValueError:
        return None
    if any(abs(value) > 10_000 for value in coefficients):
        return None
    return coefficients


def _reported_discriminant_value(tail: str) -> int | None:
    match = _FINAL_INTEGER.search(tail)
    if not match:
        return None
    return int(match.group(1))


def validate_discriminant_display_correction(
    problem: str,
    original_step: str,
    corrected_step: str,
) -> bool:
    """Validate one signed-b substitution using bounded integer arithmetic only."""
    coefficients = _quadratic_coefficients(problem)
    source = _SOURCE_DISCRIMINANT.fullmatch(original_step)
    corrected = _CORRECTED_DISCRIMINANT.fullmatch(corrected_step)
    if coefficients is None or source is None or corrected is None:
        return False
    a, b, c = coefficients
    try:
        source_values = tuple(int(source.group(key)) for key in ("b", "a", "c"))
        corrected_values = tuple(int(corrected.group(key)) for key in ("b", "a", "c"))
    except ValueError:
        return False
    if source_values != corrected_values or source_values != (b, a, c) or b >= 0:
        return False
    discriminant = b * b - 4 * a * c
    if abs(discriminant) > 1_000_000_000_000:
        return False
    return (
        _reported_discriminant_value(source.group("tail")) == discriminant
        and _reported_discriminant_value(corrected.group("tail")) == discriminant
    )


def _prepare_hint_steps(
    problem: str,
    solution_steps: str,
    stable_index: int | None,
    problem_hash: str | None,
) -> tuple[List[str], set[int]]:
    steps = split_solution_steps(solution_steps)
    corrections = load_hint_display_corrections()
    record = corrections.get(str(stable_index)) if stable_index is not None else None
    if record is None and stable_index is None and problem_hash:
        record = next(
            (entry for entry in corrections.values() if entry["problem_hash"] == str(problem_hash)),
            None,
        )

    withheld: set[int] = set()
    if record is not None:
        step_index = int(record["step_number"]) - 1
        if step_index < 0 or step_index >= len(steps):
            withheld.add(step_index)
        else:
            source_step = steps[step_index]
            source_hash = hashlib.sha256(source_step.encode("utf-8")).hexdigest()
            source_matches = (
                (stable_index is None or str(stable_index) == record["stable_index"])
                and str(problem_hash or "") == record["problem_hash"]
                and source_step == record["original_step"]
                and source_hash == record["source_step_sha256"]
            )
            arithmetic_matches = validate_discriminant_display_correction(
                problem, record["original_step"], record["corrected_step"],
            )
            if source_matches and arithmetic_matches:
                steps[step_index] = record["corrected_step"]
            else:
                withheld.add(step_index)

    suspicious = {
        index for index, step in enumerate(steps) if _SUSPICIOUS_NEGATIVE_SQUARE.search(step)
    }
    for index in suspicious:
        if not (record is not None and index == int(record["step_number"]) - 1 and index not in withheld):
            withheld.add(index)
    return steps, withheld


def choose_hint(
    problem: str,
    solution_steps: str,
    mastery: float,
    hints_used: int,
    stable_index: int | None = None,
    problem_hash: str | None = None,
) -> Dict:
    request_number = max(int(hints_used), 1)
    if request_number == 1:
        hint_type = "guided"
        if re.search(r"f\(x\)\s*=\s*4x\s*\+\s*12.*f\(0\)", str(problem), re.IGNORECASE):
            hint = "Înlocuiește x cu valoarea cerută în expresia funcției, apoi respectă ordinea operațiilor. Calculează și adună termenii."
        else:
            hint = "Notează datele cunoscute și ce trebuie aflat. Alege relația potrivită și fă primul pas fără să sari direct la rezultat."
        shown = 0
    else:
        steps, withheld = _prepare_hint_steps(problem, solution_steps, stable_index, problem_hash)
        step_index = request_number - 2
        if step_index in withheld:
            return {
                "hint_type": "review_required",
                "hint": HINT_REVIEW_MESSAGE,
                "shown_solution_steps": min(step_index, len(steps)),
            }
        if steps and 0 <= step_index < len(steps):
            is_final = step_index == len(steps) - 1
            hint_type = "soluție completă" if is_final else "pas ghidat"
            prefix = "Ultimul pas al soluției:" if is_final else f"Pasul {step_index + 1}:"
            hint = f"{prefix} {steps[step_index]}"
            shown = step_index + 1
        else:
            hint_type = "guided"
            hint = "Nu există alți pași salvați pentru acest exercițiu. Încearcă următorul pas folosind datele problemei."
            shown = len(steps)
    return {"hint_type": hint_type, "hint": hint, "shown_solution_steps": shown}


def diagnose_learning_state(correct: bool, hints_used: int, attempts: int, time_seconds: int) -> Dict:
    if correct and hints_used == 0 and attempts <= 1:
        state = "progres stabil"
        intervention = "crește dificultatea sau treci la transfer într-un context nou"
    elif correct and hints_used > 0:
        state = "înțelegere în formare"
        intervention = "dă o problemă izomorfă fără indiciu pentru verificare"
    elif not correct and hints_used >= 2:
        state = "blocaj conceptual"
        intervention = "revino la fundament și oferă succes rapid ghidat"
    elif not correct and attempts >= 2:
        state = "eroare procedurală repetată"
        intervention = "exercițiu țintit pe tipul de pas greșit"
    elif time_seconds < 30 and not correct:
        state = "stil impulsiv"
        intervention = "întrebare de conștientizare înainte de calcul"
    else:
        state = "oscilare"
        intervention = "menține dificultatea și cere justificarea strategiei"
    return {"state": state, "intervention": intervention}


def update_mastery(mastery: float, correct: bool, hints_used: int, attempts: int) -> float:
    mastery = float(np.clip(mastery, 0.05, 0.95))
    evidence = 0.16 if correct else -0.14
    evidence -= 0.04 * max(hints_used, 0)
    evidence -= 0.03 * max(attempts - 1, 0)
    return float(np.clip(mastery + evidence, 0.05, 0.97))


def review_interval_days(mastery: float, correct: bool) -> int:
    if not correct or mastery < 0.45:
        return 2
    if mastery < 0.65:
        return 7
    if mastery < 0.85:
        return 30
    return 90


def target_difficulty_from_mastery(mastery: float, correct: bool) -> str:
    if not correct or mastery < 0.35:
        return "1 - bază"
    if mastery < 0.60:
        return "2 - mediu"
    if mastery < 0.80:
        return "3 - consolidare"
    return "4 - avansat"


def generate_diagnostic_bank(data: pd.DataFrame, n_questions: int = 5, random_state: int = 42) -> List[Dict]:
    """Build a varied diagnostic set from sensible, non-empty exercise rows."""
    if n_questions <= 0 or data.empty:
        return []
    candidates = automatically_gradable_bank(data)
    candidates = candidates[candidates["Domeniu"].notna() & (candidates["Domeniu"] != "Altele")]
    valid_markers = {"nan", "none", "null", "<na>"}
    problem_text = candidates["Problema"].fillna("").astype(str).str.strip()
    answer_text = candidates["Raspunsul"].fillna("").astype(str).str.strip()
    candidates = candidates[
        problem_text.ne("") & ~problem_text.str.casefold().isin(valid_markers)
        & answer_text.ne("") & ~answer_text.str.casefold().isin(valid_markers)
    ]
    candidates = candidates.drop_duplicates(subset=["Problema", "Domeniu", "Tema_norm"], keep="first")

    if len(candidates) == 0:
        return []

    rng = np.random.default_rng(random_state)
    domains = list(candidates["Domeniu"].drop_duplicates())
    rng.shuffle(domains)
    groups = {
        domain: candidates[candidates["Domeniu"] == domain].sample(
            frac=1, random_state=(random_state + sum(ord(ch) for ch in str(domain))) % (2**32 - 1)
        )
        for domain in domains
    }
    bank = []
    max_rows = max(len(group) for group in groups.values())
    for row_number in range(max_rows):
        for domain in domains:
            group = groups[domain]
            if row_number < len(group):
                bank.append(group.iloc[row_number].to_dict())
                if len(bank) >= n_questions:
                    return bank
    return bank


def assess_diagnostic_results(responses: List[Dict], data: pd.DataFrame) -> Dict:
    """Score only verified evidence while reporting every content-eligible domain.

    Legacy prevalidated items containing a boolean ``correct`` and a domain are
    accepted when both ``verified`` and ``status`` are absent. Explicit
    verification/status fields must indicate a definite correct or incorrect
    answer.
    """
    eligible_data = automatically_gradable_bank(data)
    scores = {}
    if "Domeniu" in data:
        for domain in sorted(data["Domeniu"].dropna().unique().tolist()):
            scores[domain] = {"correct": 0, "total": 0}

    for item in responses:
        verified = item.get("verified")
        status = item.get("status")
        if "verified" in item and verified is not True:
            continue
        if status is not None and status not in {"correct", "incorrect"}:
            continue
        if "verified" not in item and status is None:
            # Legacy prevalidated responses require an explicit boolean verdict.
            pass
        correct_value = item.get("correct")
        if not isinstance(correct_value, (bool, np.bool_)):
            continue
        domain = item.get("domain") or item.get("Domeniu") or "Alt"
        correct = bool(correct_value)
        scores.setdefault(domain, {"correct": 0, "total": 0})
        scores[domain]["correct"] += int(correct)
        scores[domain]["total"] += 1

    domain_scores = {}
    for domain, stats in scores.items():
        ratio = stats["correct"] / stats["total"] if stats["total"] else None
        domain_scores[domain] = {
            "accuracy": round(ratio, 2) if ratio is not None else None,
            "correct": stats["correct"],
            "total": stats["total"],
            "priority": round(1.0 - ratio, 2) if ratio is not None else None,
            "status": "tested" if ratio is not None else "untested",
        }

    weak_domains = [
        domain for domain, stats in sorted(
            domain_scores.items(),
            key=lambda item: (
                item[1]["total"] == 0,
                -(
                    (scores[item[0]]["total"] - scores[item[0]]["correct"])
                    / scores[item[0]]["total"]
                    if scores[item[0]]["total"]
                    else 0
                ),
                str(item[0]),
            ),
        )
        if stats["total"] >= 1 and scores[domain]["correct"] / scores[domain]["total"] < 0.75
    ]

    recommended_themes = []
    for domain in weak_domains[:3]:
        if not {"Domeniu", "Tema_norm"}.issubset(eligible_data.columns):
            continue
        theme_pool = eligible_data[
            (eligible_data["Domeniu"] == domain) & (eligible_data["Tema_norm"].notna())
        ].copy()
        if len(theme_pool):
            theme_pool = theme_pool.sort_values(["Dificultate", "problem_words"], ascending=[True, True])
            recommended_themes.append({"domain": domain, "themes": theme_pool["Tema_norm"].dropna().unique().tolist()[:5]})

    return {
        "weak_domains": weak_domains,
        "domain_scores": domain_scores,
        "recommended_themes": recommended_themes,
    }


def recommend_next_exercise(
    data: pd.DataFrame,
    domain: str,
    target_difficulty: str,
    exclude_problem: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a single-row DataFrame with the recommended exercise, or an empty DataFrame.

    Returning a DataFrame keeps the return type consistent for callers in the Streamlit app.
    """
    candidates = automatically_gradable_bank(data)
    if "problem_hash" in candidates.columns:
        review_flags = Path(__file__).resolve().parents[1] / "data" / "processed" / "exercise_review_flags.csv"
        excluded_hashes = load_review_flag_hashes(review_flags)
        candidates = candidates[~candidates["problem_hash"].fillna("").astype(str).str.strip().isin(excluded_hashes)]
    if domain:
        candidates = candidates[candidates["Domeniu"] == domain]
    if exclude_problem:
        candidates = candidates[candidates["Problema"].astype(str) != str(exclude_problem)]
    if "Problema" in candidates:
        candidates = candidates[candidates["Problema"].notna() & candidates["Problema"].astype(str).str.strip().ne("")]
    if "Raspunsul" in candidates:
        candidates = candidates[candidates["Raspunsul"].notna() & candidates["Raspunsul"].astype(str).str.strip().ne("")]
    exact = candidates[candidates["Dificultate_group"] == target_difficulty]
    if len(exact) > 0:
        candidates = exact
    if len(candidates) == 0:
        return candidates.head(0)
    return candidates.sample(1, random_state=random_state)


def next_review_date(mastery: float, correct: bool) -> str:
    days = review_interval_days(mastery, correct)
    return (date.today() + timedelta(days=days)).isoformat()
