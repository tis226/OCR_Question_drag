# chunk_pdf_by_questions_gui_header_omit_table_test.py
"""
Annotate question regions in a PDF using metadata from a JSON file.
Extract explanations using a character-by-character prefix eater,
and DROP everything that appears BEFORE the LAST match of the final option (e.g., ④).

Requires:
  - PyMuPDF        (pip install pymupdf)
Optional (for table extraction):
  - pdfplumber     (pip install pdfplumber)
  - camelot-py     (pip install camelot-py[base])
  - pypdf          (pip install pypdf)
  - pandas         (pip install pandas)
"""

from __future__ import annotations
from contextlib import contextmanager, nullcontext

import re
import argparse
import json
import logging
import math
import sys
import unicodedata
import atexit
import gc
import shutil
from difflib import SequenceMatcher
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
from charwise_trace_extractor import extract_explanation_text_charwise_trace, print_trace

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit("PyMuPDF is required. Install it via 'pip install pymupdf'.") from exc

# Optional deps (used only if present)
try:
    from pypdf import PdfReader, PdfWriter  # type: ignore
except ImportError:
    PdfReader = PdfWriter = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import camelot  # type: ignore
except ImportError:
    camelot = None

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None

PDFPLUMBER_AVAILABLE = pdfplumber is not None and pd is not None
CAMEL0T_AVAILABLE = PDFPLUMBER_AVAILABLE and camelot is not None and PdfReader is not None

SOFT_HYPHEN = "\u00ad"
CONTROL_GAP_CHARS = {"\u0001"}

CIRCLED_NUMBER_CHARS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_NUMBER_PATTERN = re.compile(f"[{CIRCLED_NUMBER_CHARS}]")
CIRCLED_NUMBER_ORDER = {symbol: idx for idx, symbol in enumerate(CIRCLED_NUMBER_CHARS, start=1)}
QUESTION_NUMBER_PATTERN = re.compile(r"^\s*(\d{1,3})(?:\s*[).]|\s*번\b)")


def _rebuild_line_text(line: Dict[str, object], gap_factor: float = 0.35, min_gap: float = 0.3) -> str:
    pieces: List[str] = []
    prev_right: Optional[float] = None
    prev_width: Optional[float] = None
    spans = line.get("spans") or []
    for span in spans:
        chars = span.get("chars")
        if chars:
            for char in chars:
                glyph = char.get("c")
                if not glyph:
                    continue
                bbox = char.get("bbox") or [0, 0, 0, 0]
                width = (bbox[2] - bbox[0]) if bbox else 0.0
                if prev_right is not None and bbox:
                    gap = bbox[0] - prev_right
                    thresh = max(min_gap, (prev_width or width or min_gap) * gap_factor)
                    if gap > thresh:
                        pieces.append(" ")
                pieces.append(glyph)
                if bbox:
                    prev_right = bbox[2]
                    prev_width = width
            continue
        text = span.get("text", "")
        if text:
            pieces.append(text)
    return "".join(pieces)


def rebuild_block_text(block: Dict[str, object]) -> str:
    lines = block.get("lines") or []
    rebuilt: List[str] = []
    for line in lines:
        text = _rebuild_line_text(line)
        if text:
            rebuilt.append(text.rstrip())
    return "\n".join(rebuilt)


def normalize_text(text: str) -> str:
    """Lowercase and strip whitespace / non-letter-or-number chars for matching."""
    for ch in CONTROL_GAP_CHARS:
        text = text.replace(ch, "")
    text = unicodedata.normalize("NFKC", text.replace(SOFT_HYPHEN, ""))
    if text.startswith("<보기"):
        closing = text.find(">")
        if closing != -1:
            text = text[closing + 1 :]
    keep: List[str] = []
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category and category[0] in ("L", "N"):
            keep.append(ch.lower())
    return "".join(keep)


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for EasyOCR processing. Install it via 'pip install numpy'.")


def _require_easyocr() -> None:
    if easyocr is None:
        raise RuntimeError("EasyOCR is required for --ocr-export. Install it via 'pip install easyocr'.")


def render_pdf_to_images(
    doc: fitz.Document,
    *,
    dpi: int = 300,
    page_indices: Optional[Sequence[int]] = None,
) -> List[Tuple[int, Any]]:
    """Render PDF pages to NumPy arrays suitable for EasyOCR."""

    _require_numpy()
    zoom = max(dpi / 72.0, 1.0)
    matrix = fitz.Matrix(zoom, zoom)
    rendered: List[Tuple[int, Any]] = []
    for page_index in range(doc.page_count):
        if page_indices is not None and page_index not in page_indices:
            continue
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        buffer = pix.samples
        array = np.frombuffer(buffer, dtype=np.uint8).copy()
        array = array.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            array = array[:, :, :3]
        rendered.append((page_index, array))
    return rendered


def _bbox_from_easyocr(points: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(pt[0]) for pt in points]
    ys = [float(pt[1]) for pt in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _split_circled_options(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split leading text from circled-number options within a single OCR snippet."""

    matches = list(CIRCLED_NUMBER_PATTERN.finditer(text))
    if not matches:
        return text.strip(), []

    leading_end = matches[0].start()
    leading = text[:leading_end].strip()
    options: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        symbol = match.group(0)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        option_text = text[start:end].strip()
        options.append((symbol, option_text))
    return leading, options


def _append_option_text(options: List[Dict[str, str]], symbol: str, text: str) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    for option in options:
        if option.get("index") == symbol:
            existing = option.get("text", "").strip()
            option["text"] = f"{existing}\n{cleaned}".strip() if existing else cleaned
            return
    options.append({"index": symbol, "text": cleaned})


def group_easyocr_detections(
    detections_by_page: Sequence[Sequence[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Group EasyOCR detections into question-centric structures."""

    flattened: List[Dict[str, Any]] = []
    for page_items in detections_by_page:
        flattened.extend(page_items)

    flattened.sort(key=lambda item: (item["page_index"], item["bbox"][1], item["bbox"][0]))

    grouped: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for item in flattened:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        snippet = {
            "page_index": item["page_index"],
            "bbox": item["bbox"],
            "text": text,
            "confidence": float(item.get("confidence", 0.0)),
        }

        question_match = QUESTION_NUMBER_PATTERN.match(text)
        if question_match:
            if current:
                grouped.append(
                    {
                        "number": current["number"],
                        "text": "\n".join(part for part in current["text_parts"] if part).strip(),
                        "options": current["options"],
                        "snippets": current["snippets"],
                    }
                )
            number = int(question_match.group(1))
            remainder = text[question_match.end() :].strip()
            leading, options = _split_circled_options(remainder)
            current = {
                "number": number,
                "text_parts": [leading] if leading else [],
                "options": [],
                "snippets": [snippet],
                "options_started": False,
            }
            if options:
                current["options_started"] = True
                for symbol, option_text in options:
                    _append_option_text(current["options"], symbol, option_text)
            continue

        if current is None:
            continue

        current["snippets"].append(snippet)

        leading, options = _split_circled_options(text)
        if leading:
            if not current["options_started"]:
                current["text_parts"].append(leading)
            elif current["options"]:
                last = current["options"][-1]
                last_text = last.get("text", "").strip()
                last["text"] = f"{last_text}\n{leading}".strip() if last_text else leading
            else:
                current["text_parts"].append(leading)

        if options:
            current["options_started"] = True
            for symbol, option_text in options:
                _append_option_text(current["options"], symbol, option_text)
        elif current["options_started"] and not options and leading and current["options"]:
            # Already appended to the last option above.
            pass
        elif current["options_started"] and not options and not leading and current["options"]:
            # Continuation lines without explicit markers belong to the last option; no-op here.
            continue

    if current:
        grouped.append(
            {
                "number": current["number"],
                "text": "\n".join(part for part in current["text_parts"] if part).strip(),
                "options": current["options"],
                "snippets": current["snippets"],
            }
        )

    return grouped


def perform_easyocr_export(
    doc: fitz.Document,
    *,
    languages: Optional[Sequence[str]] = None,
    gpu: bool = False,
    dpi: int = 300,
    subject: Optional[str] = None,
    year: Optional[int] = None,
    target: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run EasyOCR on the PDF and return question-structured payloads."""

    _require_numpy()
    _require_easyocr()

    langs = list(dict.fromkeys(languages or ["ko", "en"]))
    reader = easyocr.Reader(langs, gpu=gpu)

    rendered_pages = render_pdf_to_images(doc, dpi=dpi)
    detections_by_page: List[List[Dict[str, Any]]] = []
    for page_index, image in rendered_pages:
        detections: List[Dict[str, Any]] = []
        try:
            page_results = reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("EasyOCR failed on page %s: %s", page_index + 1, exc)
            page_results = []
        for result in page_results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue
            points, text, confidence = result[:3]
            text_str = str(text or "").strip()
            if not text_str:
                continue
            bbox = _bbox_from_easyocr(points)
            detections.append(
                {
                    "page_index": page_index,
                    "bbox": bbox,
                    "text": text_str,
                    "confidence": float(confidence) if confidence is not None else 0.0,
                }
            )
        detections_by_page.append(detections)

    grouped = group_easyocr_detections(detections_by_page)

    subject_value = subject if subject is not None else None
    target_value = target if target is not None else None

    payload: List[Dict[str, Any]] = []
    for entry in grouped:
        sorted_options = sorted(
            (
                {
                    "index": option["index"],
                    "text": option.get("text", "").strip(),
                }
                for option in entry.get("options", [])
                if option.get("text")
            ),
            key=lambda opt: CIRCLED_NUMBER_ORDER.get(opt["index"], 999),
        )
        content = {
            "question_number": entry.get("number"),
            "question_text": entry.get("text", ""),
            "dispute_bool": False,
            "dispute_site": None,
            "options": sorted_options,
            "preview_image": None,
            "ocr_snippets": entry.get("snippets", []),
        }
        payload.append(
            {
                "subject": subject_value,
                "year": year,
                "target": target_value,
                "content": content,
            }
        )

    return payload


@dataclass
class Question:
    number: int
    text: str
    normalized: str
    raw_entry: dict


@dataclass
class Block:
    page_index: int
    bbox: fitz.Rect
    text: str
    normalized: str
    start: int
    end: int


@dataclass
class OmitRegion:
    page_index: int
    rect: fitz.Rect


@dataclass
class ChunkOverrideSpec:
    start_block_idx: int
    end_block_idx: int
    explanation_override: Optional[str] = None


class LinearPdfIndex:
    """Flatten a PDF into ordered text blocks with normalized lookup strings."""

    def __init__(self, doc: fitz.Document, *, omit_regions: Optional[Sequence[OmitRegion]] = None) -> None:
        self.blocks: List[Block] = []
        self._global_norm_parts: List[str] = []
        self._starts: List[int] = []
        self.global_normalized: str = ""
        self._omit_by_page: Dict[int, List[fitz.Rect]] = defaultdict(list)
        if omit_regions:
            for region in omit_regions:
                self._omit_by_page[region.page_index].append(region.rect)
        self._linearize(doc)

    def _linearize(self, doc: fitz.Document) -> None:
        cursor = 0
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw_page = page.get_text("rawdict") or {}
            raw_blocks = raw_page.get("blocks", [])
            sorted_blocks = sorted(
                (blk for blk in raw_blocks if blk.get("type") == 0),
                key=lambda b: (round(b.get("bbox", [0, 0, 0, 0])[1], 3), round(b.get("bbox", [0, 0, 0, 0])[0], 3)),
            )
            for raw in sorted_blocks:
                bbox = raw.get("bbox") or [0, 0, 0, 0]
                if len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[:4]
                raw_text = rebuild_block_text(raw)
                block_type = raw.get("type", 0)
                if block_type != 0:
                    continue  # skip images etc.
                rect = fitz.Rect(x0, y0, x1, y1)
                omit_regions = self._omit_by_page.get(page_index)
                if omit_regions and any(rect.intersects(omit) for omit in omit_regions):
                    logging.debug("Skipping block on page %s due to omit region overlap", page_index + 1)
                    continue
                cleaned = "\n".join(line.rstrip() for line in (raw_text or "").splitlines()).strip()
                if CONTROL_GAP_CHARS:
                    for bad in CONTROL_GAP_CHARS:
                        cleaned = cleaned.replace(bad, "")
                if not cleaned:
                    continue
                normalized = normalize_text(cleaned)
                if not normalized:
                    continue
                start = cursor
                cursor += len(normalized)
                block = Block(
                    page_index=page_index,
                    bbox=rect,
                    text=cleaned,
                    normalized=normalized,
                    start=start,
                    end=cursor,
                )
                self.blocks.append(block)
                self._global_norm_parts.append(normalized)
                self._starts.append(start)
        self.global_normalized = "".join(self._global_norm_parts)
        logging.debug("Indexed %s text blocks", len(self.blocks))

    def position_to_block(self, position: int) -> Optional[int]:
        idx = bisect_right(self._starts, position) - 1
        if idx < 0 or idx >= len(self.blocks):
            return None
        block = self.blocks[idx]
        return idx if position < block.end else None

    def dump_text(self, dest: Path) -> None:
        with dest.open("w", encoding="utf-8") as handle:
            current_page = -1
            for block in self.blocks:
                if block.page_index != current_page:
                    current_page = block.page_index
                    handle.write(f"\n=== Page {current_page + 1} ===\n")
                handle.write(block.text.rstrip() + "\n")


def load_chunk_overrides(path: Path) -> Dict[int, ChunkOverrideSpec]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to read chunk overrides from %s: %s", path, exc)
        return {}
    cleaned: Dict[int, ChunkOverrideSpec] = {}
    for key, spec in (data.items() if isinstance(data, dict) else []):
        try:
            qnum = int(key)
        except (TypeError, ValueError):
            logging.debug("Skipping override with non-integer key: %r", key)
            continue
        if not isinstance(spec, dict):
            logging.debug("Skipping override for Q%s (expected dict, got %r)", qnum, type(spec).__name__)
            continue
        start = spec.get("start_block_idx")
        end = spec.get("end_block_idx")
        if not isinstance(start, int) or not isinstance(end, int):
            logging.debug("Skipping override for Q%s (missing integer start/end)", qnum)
            continue
        explanation = spec.get("explanation_override")
        if isinstance(explanation, str):
            explanation = explanation.strip() or None
        else:
            explanation = None
        cleaned[qnum] = ChunkOverrideSpec(start_block_idx=start, end_block_idx=end, explanation_override=explanation)
    logging.info("Loaded %s chunk overrides from %s", len(cleaned), path)
    return cleaned


def save_chunk_overrides(path: Path, overrides: Dict[int, ChunkOverrideSpec]) -> None:
    serializable = {}
    for qnum, spec in overrides.items():
        payload = {
            "start_block_idx": spec.start_block_idx,
            "end_block_idx": spec.end_block_idx,
        }
        if spec.explanation_override:
            payload["explanation_override"] = spec.explanation_override
        serializable[str(qnum)] = payload
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote %s chunk overrides to %s", len(overrides), path)
    except OSError as exc:
        logging.error("Failed to write chunk overrides to %s: %s", path, exc)


def apply_chunk_overrides(matches: Sequence[MatchResult], overrides: Dict[int, ChunkOverrideSpec], total_blocks: int) -> None:
    if not overrides or not matches or total_blocks <= 0:
        return
    for match in matches:
        spec = overrides.get(match.question.number)
        if spec is None:
            continue
        start_raw = spec.start_block_idx
        end_raw = spec.end_block_idx
        start = max(0, min(total_blocks - 1, start_raw))
        end = max(start, min(total_blocks - 1, end_raw))
        if start != match.start_block_idx or end != match.end_block_idx:
            logging.debug(
                "Applying override for Q%s: [%s, %s] -> [%s, %s]",
                match.question.number,
                match.start_block_idx,
                match.end_block_idx,
                start,
                end,
            )
            match.start_block_idx = start
            match.end_block_idx = end
        match.manual_explanation = spec.explanation_override


def load_questions(
    json_path: Path,
    *,
    subject: Optional[str],
    year: Optional[int],
    target: Optional[str],
    only_numbers: Optional[Sequence[int]],
) -> List[Question]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    questions: List[Question] = []
    filters_applied = []

    if subject:
        filters_applied.append(f"subject={subject}")
    if year is not None:
        filters_applied.append(f"year={year}")
    if target:
        filters_applied.append(f"target={target}")
    if only_numbers:
        filters_applied.append(f"numbers={sorted(only_numbers)}")

    logging.info("Loading questions from %s%s", json_path, f" ({', '.join(filters_applied)})" if filters_applied else "")
    number_filter = set(only_numbers) if only_numbers else None

    for entry in payload:
        if subject and entry.get("subject") != subject:
            continue
        if year is not None and entry.get("year") != year:
            continue
        if target and entry.get("target") != target:
            continue

        content = entry.get("content") or {}
        number = content.get("question_number")
        text = content.get("question_text")
        if number is None or text is None:
            continue
        try:
            number_int = int(number)
        except (TypeError, ValueError):
            logging.warning("Skipping question with non-integer number: %r", number)
            continue
        if number_filter and number_int not in number_filter:
            continue

        normalized = normalize_text(text)
        if not normalized:
            logging.warning("Skipping question %s (empty after normalization)", number)
            continue

        questions.append(Question(number=number_int, text=text, normalized=normalized, raw_entry=entry))

    questions.sort(key=lambda q: q.number)
    logging.info("Loaded %s questions", len(questions))
    return questions


@dataclass
class MatchResult:
    question: Question
    start_pos: int
    start_block_idx: int
    end_block_idx: int
    matched_length: int
    partial: bool
    manually_added: bool = False
    manual_explanation: Optional[str] = None


@dataclass
class MismatchDetail:
    question_number: int
    reason: str
    normalized_length: int
    closest_page: Optional[int]
    similarity: Optional[float]
    snippet: Optional[str]
    missing_words: List[str]


def find_match_position(
    normalized: str,
    text_stream: str,
    *,
    start_offset: int,
    min_prefix_ratio: float,
) -> tuple[int, int, bool]:
    """Return (hit_index, matched_length, partial_flag) with optional prefix matching."""
    if not normalized:
        return -1, 0, False

    hit = text_stream.find(normalized, start_offset)
    if hit != -1:
        return hit, len(normalized), False

    hit = text_stream.find(normalized)
    if hit != -1:
        return hit, len(normalized), False

    prefix_len = max(1, math.ceil(len(normalized) * min_prefix_ratio))
    prefix = normalized[:prefix_len]

    hit = text_stream.find(prefix, start_offset)
    if hit == -1:
        hit = text_stream.find(prefix)

    if hit == -1:
        return -1, 0, False

    return hit, prefix_len, True


def diagnose_no_match(question: Question, index: LinearPdfIndex, *, reason: str) -> MismatchDetail:
    if not index.blocks:
        return MismatchDetail(
            question_number=question.number,
            reason=reason,
            normalized_length=len(question.normalized),
            closest_page=None,
            similarity=None,
            snippet=None,
            missing_words=[],
        )

    best_ratio = 0.0
    best_block: Optional[Block] = None
    for block in index.blocks:
        ratio = SequenceMatcher(None, question.text.lower(), block.text.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = block

    snippet = None
    if best_block:
        snippet = " ".join(best_block.text.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."

    missing_words = sorted(set(question.text.lower().split()) - set((best_block.text if best_block else "").lower().split()))
    return MismatchDetail(
        question_number=question.number,
        reason=reason,
        normalized_length=len(question.normalized),
        closest_page=(best_block.page_index + 1) if best_block else None,
        similarity=best_ratio if best_block else None,
        snippet=snippet,
        missing_words=missing_words,
    )


def match_questions_to_blocks(
    index: LinearPdfIndex,
    questions: Sequence[Question],
    *,
    min_forward_offset: int = 0,
) -> Tuple[List[MatchResult], List[MismatchDetail]]:
    matches: List[MatchResult] = []
    search_cursor = min_forward_offset
    mismatches: List[MismatchDetail] = []
    previous_context_added: set[int] = set()

    def add_previous_context(current_idx: int) -> None:
        if current_idx <= 0:
            return
        prev_question = questions[current_idx - 1]
        if prev_question.number in previous_context_added:
            return
        mismatches.append(
            MismatchDetail(
                question_number=prev_question.number,
                reason="previous_of_mismatch",
                normalized_length=len(prev_question.normalized),
                closest_page=None,
                similarity=None,
                snippet=None,
                missing_words=[],
            )
        )
        previous_context_added.add(prev_question.number)

    for idx, question in enumerate(questions):
        normalized = question.normalized
        if not normalized:
            mismatches.append(
                MismatchDetail(
                    question_number=question.number,
                    reason="empty_normalized_text",
                    normalized_length=0,
                    closest_page=None,
                    similarity=None,
                    snippet=None,
                    missing_words=[],
                )
            )
            add_previous_context(idx)
            continue

        hit, matched_len, partial = find_match_position(
            normalized,
            index.global_normalized,
            start_offset=search_cursor,
            min_prefix_ratio=0.2,
        )
        if hit == -1:
            mismatches.append(diagnose_no_match(question, index, reason="not_found_in_stream"))
            add_previous_context(idx)
            continue

        block_idx = index.position_to_block(hit)
        if block_idx is None:
            mismatches.append(diagnose_no_match(question, index, reason="no_enclosing_block"))
            add_previous_context(idx)
            continue

        matches.append(
            MatchResult(
                question=question,
                start_pos=hit,
                start_block_idx=block_idx,
                end_block_idx=block_idx,  # placeholder; patched below
                matched_length=matched_len,
                partial=partial,
            )
        )
        search_cursor = hit + matched_len

    total_blocks = len(index.blocks)
    for pos, match in enumerate(matches):
        if pos + 1 < len(matches):
            next_start_block = matches[pos + 1].start_block_idx
            match.end_block_idx = max(match.start_block_idx, next_start_block - 1)
        else:
            match.end_block_idx = total_blocks - 1

    logging.info("Matched %s/%s questions", len(matches), len(questions))
    return matches, mismatches


# ------------------ Char-by-char prefix “eater” (ONLY MODE) ------------------

def _is_wordlike(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat and cat[0] in ("L", "N")


def _match_option_prefix_charwise(
    line: str,
    candidate: str,
    *,
    max_mismatches: int = 2,
    max_lead: int = 6,
) -> Optional[int]:
    """
    Try to consume 'candidate' from the start of 'line', char-by-char, forgiving:
      - ignore whitespace and light punctuation asymmetrically,
      - allow up to 'max_mismatches' wordlike mismatches,
      - allow up to 'max_lead' leading non-word chars on the line.
    Return index in 'line' where the match ENDS if fully consumed; else None.
    """
    i = 0
    j = 0
    mismatches = 0
    nL, nC = len(line), len(candidate)

    # allow small non-wordy lead (e.g., bullets, choice symbols)
    while i < nL and not _is_wordlike(line[i]) and i < max_lead:
        i += 1

    while i < nL and j < nC:
        li = line[i]
        cj = candidate[j]
        li_cf = li.casefold()
        cj_cf = cj.casefold()

        # ignore whitespace
        if li_cf.isspace() and not cj_cf.isspace():
            i += 1
            continue
        if cj_cf.isspace() and not li_cf.isspace():
            j += 1
            continue
        if li_cf.isspace() and cj_cf.isspace():
            i += 1
            j += 1
            continue

        # ignore light punctuation asymmetrically
        if not _is_wordlike(li_cf) and _is_wordlike(cj_cf):
            i += 1
            continue
        if not _is_wordlike(cj_cf) and _is_wordlike(li_cf):
            j += 1
            continue
        if not _is_wordlike(li_cf) and not _is_wordlike(cj_cf):
            i += 1
            j += 1
            continue

        # both wordlike now
        if li_cf == cj_cf:
            i += 1
            j += 1
            continue

        mismatches += 1
        if mismatches > max_mismatches:
            return None
        i += 1  # assume OCR glitch on line; keep candidate char

    if j >= nC:
        # trim trailing non-word chars after match
        while i < nL and not _is_wordlike(line[i]):
            i += 1
        return i
    return None


def _squeeze(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def _alnum_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def extract_explanation_text_charwise_only(
    match: MatchResult,
    index: LinearPdfIndex,
    *,
    skip_texts: Optional[Iterable[str]] = None,
    max_mismatches: int = 2,
    max_lead: int = 6,
    cut_before_last_final_option: bool = True,
) -> str:
    """
    Character-by-character extractor with 'final option' hard cut:
      1) Identify the final option (last one, e.g., ④ and its text).
      2) Find the LAST line where that final option is matched (charwise, forgiving).
         - If found, DROP everything that comes before that match.
         - If not found, we won't make the hard cut.
      3) From the cut point onward:
         - If a line starts with any option text (with/without symbol), consume it and keep the suffix.
         - Drop lines that exactly equal table cells (after light normalization).
         - Keep other lines as-is.
    """
    blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
    if not blocks:
        return ""

    content = match.question.raw_entry.get("content") or {}
    option_symbols: List[str] = []
    option_texts: List[str] = []
    for opt in content.get("options") or []:
        sym = (opt.get("index") or "").strip()
        if sym and sym not in option_symbols:
            option_symbols.append(sym)
        txt = opt.get("text")
        if isinstance(txt, str):
            option_texts.append(txt)

    # Build candidates for ALL options (used for prefix-eating after the cut).
    all_candidates: List[str] = []
    for txt in option_texts:
        if isinstance(txt, str):
            all_candidates.append(txt)
    for sym, txt in zip(option_symbols, option_texts):
        if sym and isinstance(txt, str):
            all_candidates.append(f"{sym} {txt}")

    # Final option (e.g., option #4)
    final_candidates: List[str] = []
    if option_texts:
        final_txt = option_texts[-1]
        final_sym = option_symbols[-1] if option_symbols else ""
        if isinstance(final_txt, str):
            final_candidates.append(final_txt)
            if final_sym:
                final_candidates.append(f"{final_sym} {final_txt}")

    # Exact table cells to drop
    skip_texts = [t for t in (skip_texts or []) if t]
    skip_set_squeezed = {_squeeze(t) for t in skip_texts}
    skip_set_alnum = {_alnum_only(t) for t in skip_texts}

    # Collect lines across blocks
    lines: List[str] = []
    for blk in blocks:
        lines.extend(blk.text.splitlines())

    # 1) Hard-cut position: find LAST match of the final option
    cut_line_idx: int = -1
    cut_col_idx: int = -1
    if cut_before_last_final_option and final_candidates:
        for i, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            # Try all final-option shapes, pick the furthest (largest column)
            best_end = None
            for cand in final_candidates:
                res = _match_option_prefix_charwise(
                    stripped, cand, max_mismatches=max_mismatches, max_lead=max_lead
                )
                if res is not None:
                    if best_end is None or res > best_end:
                        best_end = res
            if best_end is not None:
                # We take the last such occurrence (so overwrite every time)
                cut_line_idx = i
                cut_col_idx = best_end

    kept: List[str] = []

    # Helper to decide if a normalized line equals any table cell
    def _is_exact_table_line(s: str) -> bool:
        ss = _squeeze(s)
        sa = _alnum_only(s)
        return ss in skip_set_squeezed or sa in skip_set_alnum

    # 2) Build output from the cut point onward
    start_i = 0
    if cut_line_idx >= 0:
        start_i = cut_line_idx  # we will process this line specially (cut from col)
    # else: no hard cut; keep from beginning

    for i in range(start_i, len(lines)):
        stripped = lines[i].strip()

        # First line if we have a column cut
        if i == cut_line_idx and cut_col_idx >= 0:
            suffix = stripped[cut_col_idx:].lstrip()
            if suffix and not _is_exact_table_line(suffix):
                kept.append(suffix)
            # then continue to next lines
            continue

        if not stripped:
            kept.append("")
            continue

        if _is_exact_table_line(stripped):
            continue

        # Try to consume ANY option prefix (handles stray option echoes)
        best_cut = None
        for cand in all_candidates:
            res = _match_option_prefix_charwise(
                stripped, cand, max_mismatches=max_mismatches, max_lead=max_lead
            )
            if res is not None:
                if best_cut is None or res > best_cut:
                    best_cut = res
        if best_cut is not None and best_cut < len(stripped):
            suffix = stripped[best_cut:].lstrip()
            if suffix and not _is_exact_table_line(suffix):
                kept.append(suffix)
            continue

        kept.append(lines[i].rstrip())

    # Trim outer blank lines
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()

    return "\n".join(kept)


# ------------------ (Optional) table helpers ------------------

def camelot_table_to_rows(table) -> List[List[str]]:
    if pd is None:
        return [[(" ".join(str(cell).split()) if cell else "") for cell in row] for row in table.data]
    df = table.df.copy()
    df = df.applymap(lambda value: " ".join(str(value).split()) if pd.notna(value) else "")
    return df.values.tolist()


def bbox_pp_to_camelot(bbox: Tuple[float, float, float, float], page_height: float) -> str:
    x0, top, x1, bottom = bbox
    y_top = page_height - top
    y_bottom = page_height - bottom
    return f"{x0},{y_top},{x1},{y_bottom}"


def read_camelot_tables(pdf_path: Path, page_index: int, region: Optional[str]):
    if camelot is None:
        return []
    pages_str = str(page_index + 1)
    lattice_kwargs = dict(filepath=str(pdf_path), pages=pages_str, flavor="lattice", strip_text=" \n", line_scale=40)
    if region:
        lattice_kwargs["table_regions"] = [region]
    try:
        tables = camelot.read_pdf(**lattice_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        logging.debug("Camelot lattice failed on page %s: %s", page_index + 1, exc)
        tables = []
    if len(tables) == 0:
        stream_kwargs = dict(
            filepath=str(pdf_path),
            pages=pages_str,
            flavor="stream",
            strip_text=" \n",
            row_tol=10,
            column_tol=10,
        )
        if region:
            stream_kwargs["table_regions"] = [region]
        try:
            tables = camelot.read_pdf(**stream_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Camelot stream failed on page %s: %s", page_index + 1, exc)
            tables = []
    return tables


def cleanup_camelot_tables(tables: Optional[Sequence]) -> None:
    if not tables:
        return
    for table in tables:
        parser = getattr(table, "_parser", None)
        if parser is not None:
            for attr in ("fp", "_fp", "file", "f", "stream"):
                stream = getattr(parser, attr, None)
                if stream and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass
            close_method = getattr(parser, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        temp_pdf = None
        try:
            temp_pdf = table.parsing_report.get("temp_pdf")  # type: ignore[attr-defined]
        except Exception:
            temp_pdf = None
        if temp_pdf:
            temp_path = Path(temp_pdf)
            temp_dir = temp_path.parent
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                if hasattr(atexit, "_exithandlers"):
                    handlers = list(getattr(atexit, "_exithandlers", []))
                    updated = [
                        (fn, args, kwargs)
                        for (fn, args, kwargs) in handlers
                        if not (fn is shutil.rmtree and args and Path(args[0]) == temp_dir)
                    ]
                    if len(updated) != len(handlers):
                        setattr(atexit, "_exithandlers", updated)
            except Exception:
                pass
    try:
        gc.collect()
    except Exception:
        pass


# ------------------ Region/table extraction using pdfplumber (optional) ------------------

def bbox_pp_to_rows_fallback(page, bbox_pp, *, y_tol=2.0, gap_space=1.5, join_lines_with=" "):
    # A simple char-based two-column joiner (same as in your original)
    def _lines_from_chars(chars, y_tol=2.0):
        if not chars:
            return []
        chars_sorted = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
        lines, current = [], [chars_sorted[0]]
        for ch in chars_sorted[1:]:
            if abs(ch["top"] - current[-1]["top"]) <= y_tol:
                current.append(ch)
            else:
                lines.append(current)
                current = [ch]
        lines.append(current)
        for ln in lines:
            ln.sort(key=lambda c: c["x0"])
        return lines

    def _join_line_preserve_combining(line_chars, gap_space=1.5):
        s: List[str] = []
        prev_right = None
        for ch in line_chars:
            txt = ch.get("text", "")
            if not txt:
                continue
            if unicodedata.category(txt[0]) == "Mn" and s:
                s[-1] = s[-1] + txt
            else:
                if prev_right is not None and (ch["x0"] - prev_right) > gap_space:
                    s.append(" ")
                s.append(txt)
            prev_right = ch["x1"]
        return unicodedata.normalize("NFC", "".join(s)).strip()

    def _estimate_column_split(chars, fallback_split=200.0):
        xs = sorted({round(ch["x0"], 2) for ch in chars if ch.get("text", "").strip()})
        if len(xs) < 2:
            return fallback_split
        gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
        gap, idx = max(gaps, key=lambda item: item[0])
        if gap < 10:
            return fallback_split
        return (xs[idx] + xs[idx + 1]) / 2

    cropped = page.crop(bbox_pp)
    chars = [ch for ch in cropped.chars if ch.get("text", "").strip()]
    if not chars:
        return []

    split_x = _estimate_column_split(chars)
    lines = _lines_from_chars(chars, y_tol=y_tol)

    rows: List[List[str]] = []
    current_left: List[str] = []
    current_right: List[str] = []
    right_joiner = join_lines_with if join_lines_with in (" ", "\n") else " "

    def append_row():
        if current_left or current_right:
            left_text = (" ".join(current_left)).strip()
            right_text = right_joiner.join(current_right).strip()
            rows.append([left_text, right_text])
            current_left.clear()
            current_right.clear()

    for line_chars in lines:
        left_chars = [ch for ch in line_chars if ch["x0"] < split_x]
        right_chars = [ch for ch in line_chars if ch["x0"] >= split_x]
        left_text = _join_line_preserve_combining(left_chars, gap_space=gap_space) if left_chars else ""
        right_text = _join_line_preserve_combining(right_chars, gap_space=gap_space) if right_chars else ""
        left_text = left_text.strip()
        right_text = right_text.strip()
        if left_text:
            append_row()
            current_left.append(left_text)
        if right_text:
            current_right.append(right_text)

    append_row()
    return rows


def extract_rows_from_region(
    pdf_path: Path,
    page_index: int,
    base_bbox: Tuple[float, float, float, float],
    *,
    base_page=None,
    expand_left: bool = True,
    expand_right: bool = True,
    crop_page: bool = True,
    y_tol: float = 2.0,
    gap_space: float = 1.5,
    join_lines_with: str = " ",
) -> Tuple[List[List[str]], Tuple[float, float, float, float]]:
    if not PDFPLUMBER_AVAILABLE or pdfplumber is None:
        return [], base_bbox

    close_doc = False
    page = base_page
    if page is None:
        plumber_doc = pdfplumber.open(str(pdf_path))
        page = plumber_doc.pages[page_index]
        close_doc = True

    try:
        x0, top, x1, bottom = base_bbox
        if expand_left:
            x0 = 0.0
        if expand_right:
            x1 = page.width
        expanded_bbox = (x0, top, x1, bottom)

        rows: List[List[str]] = []

        # Try Camelot first
        if CAMEL0T_AVAILABLE and camelot is not None and PdfReader is not None:
            camelot_tables: List = []
            camelot_rows: Optional[List[List[str]]] = None
            if crop_page:
                with cropped_pdf(pdf_path, page_index, expanded_bbox) as cropped_path:
                    if cropped_path is not None:
                        camelot_tables = read_camelot_tables(cropped_path, 0, None)
            if not crop_page or not camelot_tables:
                region = bbox_pp_to_camelot(expanded_bbox, page.height)
                camelot_tables = read_camelot_tables(pdf_path, page_index, region)

            if len(camelot_tables) == 1:
                camelot_rows = camelot_table_to_rows(camelot_tables[0])
                if not any(any(cell for cell in row) for row in camelot_rows):
                    camelot_rows = None
            cleanup_camelot_tables(camelot_tables)
            if camelot_rows is not None:
                return camelot_rows, expanded_bbox

        # Fallback: char-based splitter
        rows = bbox_pp_to_rows_fallback(page, expanded_bbox, y_tol=y_tol, gap_space=gap_space, join_lines_with=join_lines_with)
        return rows, expanded_bbox
    finally:
        if close_doc:
            plumber_doc.close()


def load_tables_for_page(
    pdf_path: Path,
    page_index: int,
    *,
    table_opts: Dict[str, object],
) -> List[Dict[str, object]]:
    if not PDFPLUMBER_AVAILABLE or pdfplumber is None:
        return []

    extracted: List[Dict[str, object]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            page = plumber_doc.pages[page_index]
            candidate_tables = page.find_tables() or []
            for tbl in candidate_tables:
                rows, region_bbox = extract_rows_from_region(
                    pdf_path,
                    page_index,
                    tbl.bbox,
                    base_page=page,
                    expand_left=bool(table_opts.get("expand_left", True)),
                    expand_right=bool(table_opts.get("expand_right", True)),
                    crop_page=bool(table_opts.get("crop_page", True)),
                    y_tol=table_opts.get("y_tol", 2.0),
                    gap_space=table_opts.get("gap_space", 1.5),
                    join_lines_with=table_opts.get("join_lines_with", " "),
                )
                if not rows:
                    continue
                extracted.append(
                    {"page": page_index + 1, "bbox": [round(c, 2) for c in region_bbox], "rows": rows}
                )
    except Exception as exc:
        logging.debug("Failed to extract tables on page %s: %s", page_index + 1, exc)
    return extracted


def compute_union_bbox(rects: Iterable[fitz.Rect]) -> Optional[Tuple[float, float, float, float]]:
    rect_list = list(rects)
    if not rect_list:
        return None
    x0 = min(r.x0 for r in rect_list)
    y0 = min(r.y0 for r in rect_list)
    x1 = max(r.x1 for r in rect_list)
    y1 = max(r.y1 for r in rect_list)
    return (x0, y0, x1, y1)


def pad_bbox(
    bbox: Tuple[float, float, float, float],
    pad: float,
    *,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = x1 + pad
    y1 = y1 + pad
    if max_width is not None:
        x1 = min(max_width, x1)
    if max_height is not None:
        y1 = min(max_height, y1)
    return (x0, y0, x1, y1)


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


@contextmanager
def cropped_pdf(pdf_path: Path, page_index: int, bbox_pp: Tuple[float, float, float, float]):
    if PdfReader is None or PdfWriter is None:
        yield None
        return
    writer: Optional[PdfWriter] = None  # type: ignore[assignment]
    source_handle = pdf_path.open("rb")
    try:
        reader = PdfReader(source_handle)
        page = reader.pages[page_index]
        page_height = float(page.mediabox.height)

        x0, top, x1, bottom = bbox_pp
        lower_left = (x0, page_height - bottom)
        upper_right = (x1, page_height - top)

        page.cropbox.lower_left = lower_left
        page.cropbox.upper_right = upper_right
        page.mediabox.lower_left = lower_left
        page.mediabox.upper_right = upper_right

        writer = PdfWriter()
        writer.add_page(page)
    finally:
        source_handle.close()

    if writer is None:
        yield None
        return

    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        writer.write(tmp)
        tmp_path = Path(tmp.name)

    try:
        yield tmp_path
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def extract_tables_for_match(
    pdf_path: Path,
    match: MatchResult,
    index: LinearPdfIndex,
    cache: Dict[Tuple[int, bool, bool, bool], List[Dict[str, object]]],
    table_opts: Dict[str, object],
) -> List[Dict[str, object]]:
    if not PDFPLUMBER_AVAILABLE:
        return []

    blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
    if not blocks:
        return []

    per_page: Dict[int, List[fitz.Rect]] = defaultdict(list)
    for block in blocks:
        per_page[block.page_index].append(block.bbox)

    referenced: List[Dict[str, object]] = []
    for page_index, rects in per_page.items():
        cache_key = (
            page_index,
            bool(table_opts.get("expand_left", True)),
            bool(table_opts.get("expand_right", True)),
            bool(table_opts.get("crop_page", True)),
        )
        tables = cache.get(cache_key)
        if tables is None:
            tables = load_tables_for_page(pdf_path, page_index, table_opts=table_opts)
            cache[cache_key] = tables
        if not tables:
            continue
        union_bbox = compute_union_bbox(rects)
        if union_bbox is None:
            continue
        for table_info in tables:
            table_bbox = tuple(table_info["bbox"])  # type: ignore[arg-type]
            padded = pad_bbox(union_bbox, pad=20.0)
            if bbox_intersects(padded, table_bbox):  # type: ignore[arg-type]
                referenced.append(table_info)

    return referenced


def union_rectangles(rects: Iterable[fitz.Rect], padding: float, page_rect: fitz.Rect) -> fitz.Rect:
    rect_list = list(rects)
    if not rect_list:
        return fitz.Rect()

    x0 = min(r.x0 for r in rect_list)
    y0 = min(r.y0 for r in rect_list)
    x1 = max(r.x1 for r in rect_list)
    y1 = max(r.y1 for r in rect_list)

    expanded = fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding)
    return fitz.Rect(
        max(expanded.x0, page_rect.x0),
        max(expanded.y0, page_rect.y0),
        min(expanded.x1, page_rect.x1),
        min(expanded.y1, page_rect.y1),
    )


def annotate_pdf(
    doc: fitz.Document,
    index: LinearPdfIndex,
    matches: Sequence[MatchResult],
    *,
    padding: float,
    stroke_width: float,
    label_font_size: float,
    label_prefix: str,
    text_offset: float,
) -> None:
    for match in matches:
        slice_blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
        if not slice_blocks:
            continue
        per_page: defaultdict[int, List[fitz.Rect]] = defaultdict(list)
        for block in slice_blocks:
            per_page[block.page_index].append(block.bbox)

        for page_index, rects in per_page.items():
            page = doc[page_index]
            box = union_rectangles(rects, padding, page.rect)
            if box.is_empty:
                logging.debug("Empty rectangle for question %s on page %s", match.question.number, page_index + 1)
                continue

            shape = page.new_shape()
            shape.draw_rect(box)
            shape.finish(color=(1, 0, 0), width=stroke_width)
            shape.commit()

            label_text = f"{label_prefix}{match.question.number}"
            anchor_y = max(box.y0 - text_offset, 10)
            anchor = fitz.Point(box.x0, anchor_y)
            page.insert_text(anchor, label_text, fontsize=label_font_size, color=(1, 0, 0), fontname="helv")


def review_chunks_gui(
    doc: fitz.Document,
    index: LinearPdfIndex,
    matches: List[MatchResult],
    *,
    baseline_ranges: Dict[int, Tuple[int, int]],
    missing_questions: Sequence[Question],
    max_mismatches: int,
    max_lead: int,
    override_path: Optional[Path],
) -> Tuple[bool, Dict[int, ChunkOverrideSpec]]:
    if not matches and not missing_questions:
        logging.info("No questions available for chunk review.")
        return True, {}
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, simpledialog
        import tkinter.font as tkfont
    except ImportError as exc:
        logging.error("Tkinter is required for --review-chunks but is unavailable: %s", exc)
        return False, {}

    class ChunkReviewGUI:
        def __init__(
            self,
            root: "tk.Tk",
            doc: fitz.Document,
            index: LinearPdfIndex,
            matches_ref: List[MatchResult],
            missing_questions: Sequence[Question],
            override_path: Optional[Path],
        ) -> None:
            self.root = root
            self.doc = doc
            self.index = index
            self.matches: List[MatchResult] = matches_ref
            self.baseline_ranges = baseline_ranges
            self.max_mismatches = max_mismatches
            self.max_lead = max_lead
            self.override_path = override_path
            self.missing_questions: List[Question] = list(missing_questions)
            self.blocks_by_page: Dict[int, List[Tuple[int, Block]]] = defaultdict(list)
            for idx, block in enumerate(index.blocks):
                self.blocks_by_page[block.page_index].append((idx, block))
            self.total_blocks = len(index.blocks)

            self.current_question_idx = 0
            self.current_page_index: Optional[int] = None
            self.page_sequence: List[int] = []
            self.view_zoom = 1.3
            self.render_scale = 1.5
            self.display_scale = self.view_zoom * self.render_scale
            self.preview_needs_refresh = True
            self.finished = False
            self.cancelled = False
            self.suppress_combo_event = False

            self.image_cache: Optional["tk.PhotoImage"] = None
            self.canvas_image_id: Optional[int] = None
            self.overlay_ids: List[int] = []
            self.drag_start: Optional[Tuple[float, float]] = None
            self.drag_rect_id: Optional[int] = None
            self.drag_button: Optional[int] = None
            self.dragging = False
            self.ctrl_pressed = False

            self.question_labels = self._make_question_labels()
            self.zoom_min = 0.6
            self.zoom_max = 3.5

            self.status_var = tk.StringVar(value="")
            self.page_info_var = tk.StringVar(value="")
            self.preview_status_var = tk.StringVar(value="Preview not generated")
            self.adjust_mode = tk.StringVar(value="auto")
            self.boundary_var = tk.StringVar(value="")
            self.zoom_var = tk.DoubleVar(value=self.view_zoom)
            self.missing_info_var = tk.StringVar(value="")
            self.override_status_var = tk.StringVar(value="")
            self.manual_explanations: Dict[int, str] = {
                match.question.number: match.manual_explanation
                for match in self.matches
                if match.manual_explanation
            }
            self.manual_highlights: Dict[int, Set[int]] = {}
            self.highlight_blocks: Set[int] = set()
            self.latest_auto_preview: str = ""
            self.latest_trace: Optional[Dict[str, object]] = None
            self.shift_pressed = False
            self.drag_mode: str = "chunk"

            if not self.matches and self.missing_questions:
                first_missing = self.missing_questions.pop(0)
                self._insert_match_for_question(first_missing)

            self._build_ui()
            self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
            if self.matches:
                self.load_question(0)
            else:
                self.status_var.set("No chunks defined; add a missing question to begin.")

        def _build_ui(self) -> None:
            self.root.title("Question Chunk Review")
            main = ttk.Frame(self.root, padding=6)
            main.pack(fill="both", expand=True)
            self.mono_font = tkfont.Font(family="Consolas", size=11)
            self.preview_font = tkfont.Font(family="Consolas", size=11)

            canvas_frame = ttk.Frame(main)
            canvas_frame.pack(side="left", fill="both", expand=True)
            canvas_frame.columnconfigure(0, weight=1)
            canvas_frame.rowconfigure(0, weight=1)

            self.canvas = tk.Canvas(canvas_frame, background="#1c1c1c", width=680, height=960, scrollregion=(0, 0, 0, 0))
            self.canvas.grid(row=0, column=0, sticky="nsew")
            vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
            hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
            vbar.grid(row=0, column=1, sticky="ns")
            hbar.grid(row=1, column=0, sticky="ew")
            self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
            self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
            self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
            self.canvas.bind("<ButtonPress-3>", self.on_canvas_press)
            self.canvas.bind("<B3-Motion>", self.on_canvas_drag)
            self.canvas.bind("<ButtonRelease-3>", self.on_canvas_release)
            self.canvas.bind("<MouseWheel>", self.on_scroll_pages)
            self.canvas.bind("<Button-4>", self.on_scroll_pages)
            self.canvas.bind("<Button-5>", self.on_scroll_pages)
            self.root.bind_all("<KeyPress-Control_L>", self.on_ctrl_press)
            self.root.bind_all("<KeyPress-Control_R>", self.on_ctrl_press)
            self.root.bind_all("<KeyRelease-Control_L>", self.on_ctrl_release)
            self.root.bind_all("<KeyRelease-Control_R>", self.on_ctrl_release)
            self.root.bind_all("<KeyPress-Shift_L>", self.on_shift_press)
            self.root.bind_all("<KeyPress-Shift_R>", self.on_shift_press)
            self.root.bind_all("<KeyRelease-Shift_L>", self.on_shift_release)
            self.root.bind_all("<KeyRelease-Shift_R>", self.on_shift_release)

            side_container = ttk.Frame(main, padding=(10, 0))
            side_container.pack(side="right", fill="y")
            self.side_canvas = tk.Canvas(side_container, width=360, highlightthickness=0)
            side_scrollbar = ttk.Scrollbar(side_container, orient="vertical", command=self.side_canvas.yview)
            self.side_canvas.configure(yscrollcommand=side_scrollbar.set)
            self.side_canvas.pack(side="left", fill="y", expand=False)
            side_scrollbar.pack(side="right", fill="y")
            side = ttk.Frame(self.side_canvas)
            self.side_canvas.create_window((0, 0), window=side, anchor="nw")

            def _update_side_scroll(_event: tk.Event) -> None:  # type: ignore[name-defined]
                self.side_canvas.configure(scrollregion=self.side_canvas.bbox("all"))

            side.bind("<Configure>", _update_side_scroll)

            def _side_mousewheel(event: "tk.Event") -> None:
                if event.delta:
                    self.side_canvas.yview_scroll(-1 * (event.delta // 120), "units")
                elif event.num in (4, 5):
                    self.side_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

            for widget in (side, self.side_canvas):
                widget.bind("<MouseWheel>", _side_mousewheel, add="+")
                widget.bind("<Button-4>", _side_mousewheel, add="+")
                widget.bind("<Button-5>", _side_mousewheel, add="+")

            question_frame = ttk.LabelFrame(side, text="Question")
            question_frame.pack(fill="x", pady=(0, 8))
            self.question_var = tk.StringVar()
            self.question_combo = ttk.Combobox(
                question_frame,
                textvariable=self.question_var,
                state="readonly",
                values=self.question_labels,
            )
            self.question_combo.pack(fill="x", pady=2)
            self.question_combo.bind("<<ComboboxSelected>>", self.on_question_combo)

            qnav = ttk.Frame(question_frame)
            qnav.pack(fill="x", pady=2)
            ttk.Button(qnav, text="Prev", command=self.prev_question).pack(side="left", expand=True, fill="x", padx=(0, 4))
            ttk.Button(qnav, text="Next", command=self.next_question).pack(side="left", expand=True, fill="x")

            self.question_text = tk.Text(question_frame, width=38, height=7, wrap="word")
            self.question_text.pack(fill="x", pady=4)
            self.question_text.configure(state="disabled")

            workflow_frame = ttk.LabelFrame(side, text="Actions")
            workflow_frame.pack(fill="x", pady=(0, 8))
            ttk.Button(workflow_frame, text="Save overrides", command=self.on_save_overrides).pack(fill="x", pady=2)
            ttk.Button(workflow_frame, text="Apply & Continue", command=self.on_finish).pack(fill="x", pady=2)
            ttk.Button(workflow_frame, text="Drop question", command=self.drop_current_question).pack(fill="x", pady=2)
            ttk.Button(workflow_frame, text="Cancel", command=self.on_cancel).pack(fill="x", pady=2)

            page_frame = ttk.LabelFrame(side, text="Page view")
            page_frame.pack(fill="x", pady=(0, 8))
            page_nav = ttk.Frame(page_frame)
            page_nav.pack(fill="x", pady=2)
            ttk.Button(page_nav, text="Prev page", command=self.prev_page).pack(side="left", expand=True, fill="x", padx=(0, 4))
            ttk.Button(page_nav, text="Next page", command=self.next_page).pack(side="left", expand=True, fill="x")
            ttk.Label(page_frame, textvariable=self.page_info_var).pack(fill="x")

            zoom_frame = ttk.Frame(page_frame)
            zoom_frame.pack(fill="x", pady=(4, 0))
            ttk.Label(zoom_frame, text="Zoom").pack(side="left")
            zoom_scale = ttk.Scale(
                zoom_frame,
                from_=self.zoom_min,
                to=self.zoom_max,
                variable=self.zoom_var,
                command=self.on_zoom_change,
            )
            zoom_scale.pack(side="left", fill="x", expand=True, padx=6)

            adjust_frame = ttk.LabelFrame(side, text="Adjust bounds")
            adjust_frame.pack(fill="x", pady=(0, 8))
            mode_frame = ttk.Frame(adjust_frame)
            mode_frame.pack(fill="x", pady=(0, 4))
            for label, value in (("Auto click", "auto"), ("Start", "start"), ("End", "end")):
                ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.adjust_mode).pack(side="left", padx=2)

            shift_frame = ttk.Frame(adjust_frame)
            shift_frame.pack(fill="x", pady=2)
            ttk.Button(shift_frame, text="Start -", command=lambda: self.bump_boundary("start", -1)).pack(
                side="left", expand=True, fill="x", padx=(0, 2)
            )
            ttk.Button(shift_frame, text="Start +", command=lambda: self.bump_boundary("start", 1)).pack(
                side="left", expand=True, fill="x"
            )

            shift_frame2 = ttk.Frame(adjust_frame)
            shift_frame2.pack(fill="x", pady=2)
            ttk.Button(shift_frame2, text="End -", command=lambda: self.bump_boundary("end", -1)).pack(
                side="left", expand=True, fill="x", padx=(0, 2)
            )
            ttk.Button(shift_frame2, text="End +", command=lambda: self.bump_boundary("end", 1)).pack(
                side="left", expand=True, fill="x"
            )

            control_frame = ttk.Frame(adjust_frame)
            control_frame.pack(fill="x", pady=2)
            ttk.Button(control_frame, text="Reset", command=self.reset_to_baseline).pack(side="left", expand=True, fill="x", padx=(0, 2))
            ttk.Button(control_frame, text="Snap to page", command=self.snap_to_page).pack(side="left", expand=True, fill="x")

            ttk.Label(adjust_frame, textvariable=self.boundary_var, foreground="#444").pack(fill="x", pady=(4, 0))
            ttk.Button(adjust_frame, text="Add missing question", command=self.add_missing_question).pack(fill="x", pady=(4, 0))
            ttk.Label(adjust_frame, textvariable=self.missing_info_var, foreground="#666").pack(fill="x", pady=(2, 0))

            chunk_frame = ttk.LabelFrame(side, text="Chunk blocks")
            chunk_frame.pack(fill="both", expand=True, pady=(0, 8))
            self.chunk_text = tk.Text(chunk_frame, width=40, height=18, wrap="word", font=self.mono_font)
            self.chunk_text.pack(fill="both", expand=True)
            self.chunk_text.configure(state="disabled", spacing1=3, spacing3=2)

            preview_frame = ttk.LabelFrame(side, text="Extraction preview")
            preview_frame.pack(fill="both", expand=True, pady=(0, 8))
            ttk.Button(preview_frame, text="Refresh preview", command=self.refresh_preview).pack(anchor="w", pady=(0, 4))
            ttk.Label(preview_frame, textvariable=self.preview_status_var).pack(fill="x")
            self.preview_text = tk.Text(preview_frame, width=40, height=12, wrap="word", font=self.preview_font, undo=True)
            self.preview_text.pack(fill="both", expand=True)
            self.preview_text.configure(state="normal", spacing1=3, spacing3=2)
            override_btns = ttk.Frame(preview_frame)
            override_btns.pack(fill="x", pady=(4, 2))
            ttk.Button(override_btns, text="Apply override", command=self.apply_manual_override).pack(
                side="left", expand=True, fill="x", padx=(0, 4)
            )
            ttk.Button(override_btns, text="Clear override", command=self.clear_manual_override).pack(
                side="left", expand=True, fill="x"
            )
            ttk.Label(preview_frame, textvariable=self.override_status_var, foreground="#774400").pack(fill="x")

            ttk.Label(side, textvariable=self.status_var, wraplength=320, justify="left").pack(fill="x", pady=(6, 0))
            self._refresh_question_combo_values()
            self._update_missing_label()

        def run(self) -> None:
            self.root.mainloop()

        def current_match(self) -> MatchResult:
            return self.matches[self.current_question_idx]

        def _make_question_labels(self) -> List[str]:
            labels: List[str] = []
            for match in self.matches:
                suffix = "*" if match.manually_added else ""
                labels.append(f"Q{match.question.number}{suffix}")
            return labels

        def _refresh_question_combo_values(self) -> None:
            self.question_labels = self._make_question_labels()
            self.question_combo["values"] = self.question_labels
            if self.matches and self.current_question_idx < len(self.matches):
                self.suppress_combo_event = True
                try:
                    self.question_combo.current(self.current_question_idx)
                    self.question_var.set(self.question_labels[self.current_question_idx])
                finally:
                    self.suppress_combo_event = False
            else:
                self.question_var.set("")

        def _update_missing_label(self) -> None:
            if not self.missing_questions:
                self.missing_info_var.set("Missing questions: none")
            else:
                preview = ", ".join(str(q.number) for q in self.missing_questions[:10])
                suffix = "..." if len(self.missing_questions) > 10 else ""
                self.missing_info_var.set(f"Missing questions: {preview}{suffix}")

        def _default_block_index(self) -> int:
            if self.current_page_index is not None:
                page_blocks = self.blocks_by_page.get(self.current_page_index)
                if page_blocks:
                    return page_blocks[0][0]
            return max(self.total_blocks - 1, 0)

        def _insert_match_for_question(self, question: Question) -> int:
            default_idx = min(max(self._default_block_index(), 0), max(self.total_blocks - 1, 0))
            new_match = MatchResult(
                question=question,
                start_pos=-1,
                start_block_idx=default_idx,
                end_block_idx=default_idx,
                matched_length=0,
                partial=True,
                manually_added=True,
            )
            insert_idx = 0
            while insert_idx < len(self.matches) and self.matches[insert_idx].question.number < question.number:
                insert_idx += 1
            self.matches.insert(insert_idx, new_match)
            return insert_idx

        def _pages_for_match(self, match: MatchResult) -> List[int]:
            pages: List[int] = []
            seen: set[int] = set()
            for idx in range(match.start_block_idx, match.end_block_idx + 1):
                if idx < 0 or idx >= self.total_blocks:
                    continue
                block = self.index.blocks[idx]
                if block.page_index not in seen:
                    seen.add(block.page_index)
                    pages.append(block.page_index)
            if pages:
                return pages
            if self.total_blocks and 0 <= match.start_block_idx < self.total_blocks:
                return [self.index.blocks[match.start_block_idx].page_index]
            return [0]

        def load_question(self, idx: int) -> None:
            if not self.matches:
                self.current_question_idx = 0
                self.question_text.configure(state="normal")
                self.question_text.delete("1.0", "end")
                self.question_text.insert("1.0", "No chunks defined. Use 'Add missing question' to start.")
                self.question_text.configure(state="disabled")
                self.chunk_text.configure(state="normal")
                self.chunk_text.delete("1.0", "end")
                self.chunk_text.insert("1.0", "[empty]")
                self.chunk_text.configure(state="disabled")
                self.canvas.delete("all")
                self.page_info_var.set("")
                self.boundary_var.set("")
                self.set_preview_text("")
                self.preview_status_var.set("Preview unavailable")
                self.status_var.set("No chunks yet.")
                self.override_status_var.set("")
                return

            idx = max(0, min(len(self.matches) - 1, idx))
            self.current_question_idx = idx
            self._refresh_question_combo_values()
            match = self.matches[idx]
            self.question_text.configure(state="normal")
            self.question_text.delete("1.0", "end")
            question_text = match.question.text.strip()
            self.question_text.insert("1.0", question_text)
            self.question_text.configure(state="disabled")
            self.page_sequence = self._pages_for_match(match)
            self.current_page_index = self.page_sequence[0]
            self.page_info_var.set("")
            self.update_block_summary()
            self.update_boundary_label()
            self.update_status()
            manual_highlight = self.manual_highlights.get(match.question.number)
            if manual_highlight:
                self.highlight_blocks = set(manual_highlight)
                manual_text = self.manual_explanations.get(match.question.number)
                if manual_text:
                    self.set_preview_text(manual_text)
                else:
                    auto_text = self._build_text_from_blocks(manual_highlight)
                    self.set_preview_text(auto_text)
                    self.manual_explanations[match.question.number] = auto_text
                    match.manual_explanation = auto_text
                self.preview_status_var.set("Preview (highlight override)")
            else:
                self.update_explanation_metadata(update_preview=False)
                manual = self.manual_explanations.get(match.question.number)
                if manual:
                    self.set_preview_text(manual)
                    self.preview_status_var.set("Preview (manual override)")
                else:
                    self.set_preview_text(self.latest_auto_preview or "")
            self.update_override_status()
            self.render_page()

        def next_question(self) -> None:
            self.load_question(min(len(self.matches) - 1, self.current_question_idx + 1))

        def prev_question(self) -> None:
            self.load_question(max(0, self.current_question_idx - 1))

        def on_question_combo(self, _event: object) -> None:
            if self.suppress_combo_event:
                return
            try:
                idx = self.question_labels.index(self.question_var.get())
            except ValueError:
                return
            self.load_question(idx)

        def _ensure_page_valid(self, match: MatchResult) -> None:
            pages = self._pages_for_match(match)
            self.page_sequence = pages
            if self.current_page_index not in pages:
                self.current_page_index = pages[0]

        def next_page(self) -> None:
            if not self.page_sequence:
                return
            try:
                current_pos = self.page_sequence.index(self.current_page_index)
            except ValueError:
                self.current_page_index = self.page_sequence[0]
                current_pos = 0
            new_pos = min(len(self.page_sequence) - 1, current_pos + 1)
            if new_pos != current_pos:
                self.current_page_index = self.page_sequence[new_pos]
                self.render_page()

        def prev_page(self) -> None:
            if not self.page_sequence:
                return
            try:
                current_pos = self.page_sequence.index(self.current_page_index)
            except ValueError:
                self.current_page_index = self.page_sequence[0]
                current_pos = 0
            new_pos = max(0, current_pos - 1)
            if new_pos != current_pos:
                self.current_page_index = self.page_sequence[new_pos]
                self.render_page()

        def on_zoom_change(self, value: str) -> None:
            try:
                self.view_zoom = float(value)
            except ValueError:
                return
            self.view_zoom = max(self.zoom_min, min(self.zoom_max, self.view_zoom))
            self.render_page()

        def on_scroll_pages(self, event: "tk.Event") -> None:
            if not self.matches:
                return
            if event.num in (4, 5):
                direction = -1 if event.num == 4 else 1
            else:
                raw = getattr(event, "delta", 0)
                direction = -1 if raw > 0 else 1 if raw < 0 else 0
            if direction == 0:
                return
            top, bottom = self.canvas.yview()
            if direction < 0:
                if top <= 0.0:
                    prev_index = self.current_page_index
                    self.prev_page()
                    if self.current_page_index != prev_index:
                        self.canvas.yview_moveto(1.0)
                else:
                    self.canvas.yview_scroll(direction, "units")
            else:
                if bottom >= 1.0:
                    prev_index = self.current_page_index
                    self.next_page()
                    if self.current_page_index != prev_index:
                        self.canvas.yview_moveto(0.0)
                else:
                    self.canvas.yview_scroll(direction, "units")

        def render_page(self) -> None:
            if not self.matches:
                self.canvas.delete("all")
                self.page_info_var.set("")
                return
            match = self.current_match()
            if self.current_page_index is None:
                self._ensure_page_valid(match)
            if self.current_page_index is None:
                return
            try:
                page = self.doc[self.current_page_index]
            except Exception:
                return
            scale = self.view_zoom * self.render_scale
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            self.display_scale = scale
            data = pix.tobytes("ppm")
            self.canvas.delete("all")
            self.image_cache = tk.PhotoImage(data=data)
            self.canvas_image_id = self.canvas.create_image(0, 0, image=self.image_cache, anchor="nw")
            self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))
            self.draw_overlays()
            if self.page_sequence:
                try:
                    idx = self.page_sequence.index(self.current_page_index)
                except ValueError:
                    idx = 0
                self.page_info_var.set(
                    f"PDF page {self.current_page_index + 1} ({idx + 1}/{len(self.page_sequence)})"
                )

        def draw_overlays(self) -> None:
            if self.current_page_index is None or not self.matches:
                return
            match = self.current_match()
            baseline = self.baseline_ranges.get(match.question.number)
            baseline_start = baseline[0] if baseline else None
            baseline_end = baseline[1] if baseline else None
            selected_range = set(range(match.start_block_idx, match.end_block_idx + 1))
            for item_id in self.overlay_ids:
                self.canvas.delete(item_id)
            self.overlay_ids.clear()
            zoom = self.display_scale or self.view_zoom or 1.0
            highlight = self.highlight_blocks
            for block_idx, block in self.blocks_by_page.get(self.current_page_index, []):
                rect = block.bbox
                x0 = rect.x0 * zoom
                y0 = rect.y0 * zoom
                x1 = rect.x1 * zoom
                y1 = rect.y1 * zoom
                if block_idx in highlight:
                    fill_id = self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        outline="",
                        fill="#ff5f5f",
                        stipple="gray50",
                    )
                    self.overlay_ids.append(fill_id)
                color = "#5c5c5c"
                width = 1
                dash = None
                if block_idx == match.start_block_idx:
                    color = "#1f77b4"
                    width = 3
                elif block_idx == match.end_block_idx:
                    color = "#d62728"
                    width = 3
                elif block_idx in selected_range:
                    color = "#2ca02c"
                    width = 2
                elif baseline_start is not None and baseline_end is not None and baseline_start <= block_idx <= baseline_end:
                    color = "#c7c730"
                    dash = (4, 4)
                overlay_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width, dash=dash)
                self.overlay_ids.append(overlay_id)

        def bump_boundary(self, target: str, delta: int) -> None:
            if not self.matches:
                return
            match = self.current_match()
            if target == "start":
                new_start = max(0, min(match.start_block_idx + delta, match.end_block_idx))
                match.start_block_idx = new_start
            elif target == "end":
                new_end = min(self.total_blocks - 1, max(match.start_block_idx, match.end_block_idx + delta))
                match.end_block_idx = new_end
            else:
                return
            self._after_boundary_change()

        def reset_to_baseline(self) -> None:
            if not self.matches:
                return
            match = self.current_match()
            baseline = self.baseline_ranges.get(match.question.number)
            if not baseline:
                return
            start, end = baseline
            match.start_block_idx = max(0, min(start, self.total_blocks - 1))
            match.end_block_idx = max(match.start_block_idx, min(end, self.total_blocks - 1))
            self._after_boundary_change()

        def snap_to_page(self) -> None:
            if self.current_page_index is None or not self.matches:
                return
            page_blocks = self.blocks_by_page.get(self.current_page_index, [])
            if not page_blocks:
                return
            first_idx = page_blocks[0][0]
            last_idx = page_blocks[-1][0]
            match = self.current_match()
            match.start_block_idx = min(match.start_block_idx, first_idx)
            match.end_block_idx = max(match.end_block_idx, last_idx)
            self._after_boundary_change()

        def _after_boundary_change(self) -> None:
            if not self.matches:
                return
            match = self.current_match()
            if match.start_block_idx > match.end_block_idx:
                match.end_block_idx = match.start_block_idx
            match.start_block_idx = max(0, min(match.start_block_idx, self.total_blocks - 1))
            match.end_block_idx = max(0, min(match.end_block_idx, self.total_blocks - 1))
            self._ensure_page_valid(match)
            self.update_explanation_metadata(update_preview=False)
            manual = self.manual_explanations.get(match.question.number)
            if not manual:
                self.set_preview_text(self.latest_auto_preview or "")
            self.render_page()
            self.update_block_summary()
            self.update_boundary_label()
            self.update_status()
            self.update_override_status()

        def update_block_summary(self) -> None:
            match = self.current_match()
            lines: List[str] = []
            max_lines = 120
            for idx in range(match.start_block_idx, match.end_block_idx + 1):
                if idx < 0 or idx >= self.total_blocks:
                    continue
                block = self.index.blocks[idx]
                snippet = " ".join(block.text.strip().split())
                if len(snippet) > 140:
                    snippet = snippet[:137] + "..."
                lines.append(f"[p{block.page_index + 1:03}] #{idx}: {snippet}")
                if len(lines) >= max_lines:
                    lines.append("... (truncated)")
                    break
            text = "\n".join(lines) if lines else "[empty chunk]"
            self.chunk_text.configure(state="normal")
            self.chunk_text.delete("1.0", "end")
            self.chunk_text.insert("1.0", text)
            self.chunk_text.configure(state="disabled")

        def update_boundary_label(self) -> None:
            match = self.current_match()
            base = self.baseline_ranges.get(match.question.number)
            base_text = f"{base[0]}–{base[1]}" if base else "n/a"
            self.boundary_var.set(
                f"Current blocks: {match.start_block_idx}–{match.end_block_idx} (baseline {base_text})"
            )

        def update_status(self) -> None:
            match = self.current_match()
            base = self.baseline_ranges.get(match.question.number)
            modified = base != (match.start_block_idx, match.end_block_idx)
            pages = ", ".join(str(p + 1) for p in self.page_sequence)
            origin = "manual" if match.manually_added else "auto"
            flag = "modified" if modified else "original"
            self.status_var.set(f"Q{match.question.number} covering pages [{pages}] — {origin}, {flag}")

        def set_preview_text(self, text: str) -> None:
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", text)
            self.preview_text.edit_modified(False)

        def on_canvas_press(self, event: "tk.Event") -> None:
            if event.num not in (1, 3):
                return
            self.drag_mode = "highlight" if self.shift_pressed else "chunk"
            self.canvas.focus_set()
            self.drag_button = event.num
            self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self.dragging = False
            if self.drag_rect_id is not None:
                self.canvas.delete(self.drag_rect_id)
            outline = "#1f77b4" if event.num == 1 else "#d62728"
            self.drag_rect_id = self.canvas.create_rectangle(
                self.drag_start[0],
                self.drag_start[1],
                self.drag_start[0],
                self.drag_start[1],
                outline=outline,
                dash=(4, 2),
                width=2,
            )

        def on_canvas_drag(self, event: "tk.Event") -> None:
            if self.drag_start is None or self.drag_button not in (1, 3):
                return
            current = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            if self.drag_rect_id is None:
                outline = "#1f77b4" if self.drag_button == 1 else "#d62728"
                self.drag_rect_id = self.canvas.create_rectangle(
                    self.drag_start[0],
                    self.drag_start[1],
                    current[0],
                    current[1],
                    outline=outline,
                    dash=(4, 2),
                    width=2,
                )
            else:
                self.canvas.coords(self.drag_rect_id, self.drag_start[0], self.drag_start[1], current[0], current[1])
            if not self.dragging and (
                abs(current[0] - self.drag_start[0]) > 4 or abs(current[1] - self.drag_start[1]) > 4
            ):
                self.dragging = True

        def on_canvas_release(self, event: "tk.Event") -> None:
            if self.drag_start is None or event.num not in (1, 3):
                return
            end_point = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            start_point = self.drag_start
            dragged = self.dragging
            button = self.drag_button
            if self.drag_rect_id is not None:
                self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None
            self.drag_start = None
            self.dragging = False
            self.drag_button = None
            if not dragged:
                if self.drag_mode == "highlight":
                    self.apply_highlight_click(event, event.num)
                else:
                    self._handle_canvas_click(event)
                return
            if self.drag_mode == "highlight":
                self.apply_highlight_drag(start_point, end_point, button)
            else:
                self.apply_drag_selection(start_point, end_point, button, additive=self.ctrl_pressed)

        def apply_drag_selection(
            self,
            start_canvas: Tuple[float, float],
            end_canvas: Tuple[float, float],
            button: Optional[int],
            *,
            additive: bool,
        ) -> None:
            if self.current_page_index is None or not self.matches or button not in (1, 3):
                return
            scale = self.display_scale or self.view_zoom or 1.0
            rect = fitz.Rect(
                min(start_canvas[0], end_canvas[0]) / scale,
                min(start_canvas[1], end_canvas[1]) / scale,
                max(start_canvas[0], end_canvas[0]) / scale,
                max(start_canvas[1], end_canvas[1]) / scale,
            )
            selected_blocks = [
                idx
                for idx, block in self.blocks_by_page.get(self.current_page_index, [])
                if block.bbox.intersects(rect)
            ]
            if not selected_blocks:
                return
            sel_min = min(selected_blocks)
            sel_max = max(selected_blocks)
            if button == 1:
                self._add_selection(sel_min, sel_max, additive=additive)
            else:
                self._unselect_range(sel_min, sel_max, additive=additive)

        def apply_highlight_drag(
            self,
            start_canvas: Tuple[float, float],
            end_canvas: Tuple[float, float],
            button: Optional[int],
        ) -> None:
            if self.current_page_index is None or not self.matches or button not in (1, 3):
                return
            scale = self.display_scale or self.view_zoom or 1.0
            rect = fitz.Rect(
                min(start_canvas[0], end_canvas[0]) / scale,
                min(start_canvas[1], end_canvas[1]) / scale,
                max(start_canvas[0], end_canvas[0]) / scale,
                max(start_canvas[1], end_canvas[1]) / scale,
            )
            selected_blocks = [
                idx
                for idx, block in self.blocks_by_page.get(self.current_page_index, [])
                if block.bbox.intersects(rect)
            ]
            if not selected_blocks:
                return
            if button == 1:
                self._add_to_manual_highlight(selected_blocks)
            else:
                self._remove_from_manual_highlight(selected_blocks)

        def apply_highlight_click(self, event: "tk.Event", button: int) -> None:
            if self.current_page_index is None or not self.matches or button not in (1, 3):
                return
            scale = self.display_scale or self.view_zoom or 1.0
            x = self.canvas.canvasx(event.x) / scale
            y = self.canvas.canvasy(event.y) / scale
            block_idx = self._block_at_point(self.current_page_index, fitz.Point(x, y))
            if block_idx is None:
                return
            if button == 1:
                self._add_to_manual_highlight([block_idx])
            else:
                self._remove_from_manual_highlight([block_idx])

        def _add_to_manual_highlight(self, block_indices: Iterable[int]) -> None:
            match = self.current_match()
            block_set = self.manual_highlights.setdefault(match.question.number, set())
            for idx in block_indices:
                if 0 <= idx < self.total_blocks:
                    block_set.add(idx)
            self._apply_highlight_override(match, block_set)

        def _remove_from_manual_highlight(self, block_indices: Iterable[int]) -> None:
            match = self.current_match()
            block_set = self.manual_highlights.get(match.question.number)
            if not block_set:
                return
            for idx in block_indices:
                block_set.discard(idx)
            self._apply_highlight_override(match, block_set)

        def _apply_highlight_override(self, match: MatchResult, block_set: Set[int]) -> None:
            cleaned = {idx for idx in block_set if 0 <= idx < self.total_blocks}
            if not cleaned:
                self.manual_highlights.pop(match.question.number, None)
                self.manual_explanations.pop(match.question.number, None)
                match.manual_explanation = None
                self.override_status_var.set("Manual highlight cleared.")
                self.update_explanation_metadata(update_preview=True)
                self.update_override_status()
                return
            self.manual_highlights[match.question.number] = cleaned
            text = self._build_text_from_blocks(cleaned)
            self.manual_explanations[match.question.number] = text
            match.manual_explanation = text
            self.highlight_blocks = set(cleaned)
            self.set_preview_text(text)
            self.override_status_var.set("Manual highlight override.")
            self.preview_status_var.set("Preview (highlight override)")
            self.render_page()
            self.update_override_status()

        def _build_text_from_blocks(self, block_indices: Iterable[int]) -> str:
            lines: List[str] = []
            for idx in sorted(set(block_indices)):
                if 0 <= idx < self.total_blocks:
                    lines.append(self.index.blocks[idx].text.rstrip())
            return "\n".join(lines).strip()

        def _add_selection(self, sel_min: int, sel_max: int, *, additive: bool) -> None:
            match = self.current_match()
            if not additive:
                match.start_block_idx = min(sel_min, sel_max)
                match.end_block_idx = max(sel_min, sel_max)
                self._after_boundary_change()
                return
            current_range = range(match.start_block_idx, match.end_block_idx + 1)
            new_range = range(min(sel_min, sel_max), max(sel_min, sel_max) + 1)
            combined = sorted(set(current_range).union(set(new_range)))
            match.start_block_idx = combined[0]
            match.end_block_idx = combined[-1]
            self._after_boundary_change()

        def _unselect_range(self, sel_min: int, sel_max: int, *, additive: bool) -> None:
            match = self.current_match()
            if sel_max < match.start_block_idx or sel_min > match.end_block_idx:
                return
            overlap_start = max(match.start_block_idx, sel_min)
            overlap_end = min(match.end_block_idx, sel_max)
            if overlap_start > overlap_end:
                return
            if overlap_start == match.start_block_idx and overlap_end == match.end_block_idx and additive:
                return
            if overlap_start == match.start_block_idx:
                match.start_block_idx = min(max(overlap_end + 1, match.start_block_idx), match.end_block_idx)
            elif overlap_end == match.end_block_idx:
                match.end_block_idx = max(match.start_block_idx, overlap_start - 1)
            else:
                left_span = overlap_start - match.start_block_idx
                right_span = match.end_block_idx - overlap_end
                if additive:
                    if left_span <= right_span:
                        match.start_block_idx = min(max(overlap_end + 1, match.start_block_idx), match.end_block_idx)
                    else:
                        match.end_block_idx = max(match.start_block_idx, overlap_start - 1)
                else:
                    if left_span <= right_span:
                        match.start_block_idx = min(max(overlap_end + 1, match.start_block_idx), match.end_block_idx)
                    else:
                        match.end_block_idx = max(match.start_block_idx, overlap_start - 1)
            self._after_boundary_change()

        def _handle_canvas_click(self, event: "tk.Event") -> None:
            if self.current_page_index is None or not self.matches:
                return
            scale = self.display_scale or self.view_zoom or 1.0
            x = self.canvas.canvasx(event.x) / scale
            y = self.canvas.canvasy(event.y) / scale
            block_idx = self._block_at_point(self.current_page_index, fitz.Point(x, y))
            if block_idx is None:
                return
            mode = self.adjust_mode.get()
            match = self.current_match()
            if mode == "start":
                self._set_boundary("start", block_idx)
            elif mode == "end":
                self._set_boundary("end", block_idx)
            else:
                if block_idx <= match.start_block_idx:
                    self._set_boundary("start", block_idx)
                elif block_idx >= match.end_block_idx:
                    self._set_boundary("end", block_idx)
                else:
                    dist_start = abs(block_idx - match.start_block_idx)
                    dist_end = abs(block_idx - match.end_block_idx)
                    self._set_boundary("start" if dist_start <= dist_end else "end", block_idx)

        def _block_at_point(self, page_index: int, point: fitz.Point) -> Optional[int]:
            for block_idx, block in self.blocks_by_page.get(page_index, []):
                if block.bbox.contains(point):
                    return block_idx
            return None

        def _set_boundary(self, target: str, block_idx: int) -> None:
            match = self.current_match()
            block_idx = max(0, min(block_idx, self.total_blocks - 1))
            if target == "start":
                if block_idx > match.end_block_idx:
                    match.end_block_idx = block_idx
                match.start_block_idx = block_idx
            elif target == "end":
                if block_idx < match.start_block_idx:
                    match.start_block_idx = block_idx
                match.end_block_idx = block_idx
            self._after_boundary_change()

        def refresh_preview(self) -> None:
            if not self.matches:
                return
            self.update_explanation_metadata(update_preview=True)
            self.preview_needs_refresh = False
            self.render_page()
            self.update_override_status()

        def add_missing_question(self) -> None:
            if not self.missing_questions:
                messagebox.showinfo("Add missing question", "All questions already have chunks.")
                return
            remaining = ", ".join(str(q.number) for q in self.missing_questions[:10])
            prompt = f"Enter a question number to add.\nMissing: {remaining}"
            answer = simpledialog.askinteger("Add missing question", prompt, parent=self.root, minvalue=1)
            if answer is None:
                return
            question = next((q for q in self.missing_questions if q.number == answer), None)
            if question is None:
                messagebox.showerror("Add missing question", f"Question {answer} is not marked as missing.")
                return
            self.missing_questions = [q for q in self.missing_questions if q.number != answer]
            insert_idx = self._insert_match_for_question(question)
            self.current_question_idx = insert_idx
            self._refresh_question_combo_values()
            self._update_missing_label()
            self.load_question(insert_idx)

        def _compute_overrides(self) -> Dict[int, ChunkOverrideSpec]:
            overrides: Dict[int, ChunkOverrideSpec] = {}
            for match in self.matches:
                rng = (match.start_block_idx, match.end_block_idx)
                baseline = self.baseline_ranges.get(match.question.number)
                manual_text = self.manual_explanations.get(match.question.number)
                if baseline != rng or manual_text:
                    overrides[match.question.number] = ChunkOverrideSpec(
                        start_block_idx=rng[0],
                        end_block_idx=rng[1],
                        explanation_override=manual_text,
                    )
            return overrides

        def on_save_overrides(self) -> None:
            if not self.override_path:
                messagebox.showinfo(
                    "Save overrides", "Provide --chunk-overrides on the CLI to enable saving overrides from this dialog."
                )
                return
            overrides = self._compute_overrides()
            save_chunk_overrides(self.override_path, overrides)
            messagebox.showinfo("Save overrides", f"Saved {len(overrides)} overrides to {self.override_path}")

        def apply_manual_override(self) -> None:
            if not self.matches:
                return
            text = self.preview_text.get("1.0", "end").strip()
            if not text:
                messagebox.showerror("Apply override", "Cannot save an empty override. Clear instead if needed.")
                return
            match = self.current_match()
            self.manual_explanations[match.question.number] = text
            match.manual_explanation = text
            self.override_status_var.set("Manual override saved.")
            self.update_override_status()

        def clear_manual_override(self) -> None:
            if not self.matches:
                return
            match = self.current_match()
            removed = self.manual_explanations.pop(match.question.number, None)
            match.manual_explanation = None
            self.manual_highlights.pop(match.question.number, None)
            if removed is not None:
                self.override_status_var.set("Manual override cleared.")
            else:
                self.override_status_var.set("No manual override.")
            self.set_preview_text(self.latest_auto_preview or "")
            self.update_explanation_metadata(update_preview=False)
            self.update_override_status()

        def update_override_status(self) -> None:
            if not self.matches:
                self.override_status_var.set("")
                return
            match = self.current_match()
            if match.question.number in self.manual_explanations:
                self.override_status_var.set("Manual override active.")
            else:
                self.override_status_var.set("No manual override.")

        def update_explanation_metadata(self, *, update_preview: bool = False) -> None:
            if not self.matches:
                self.highlight_blocks = set()
                self.latest_auto_preview = ""
                self.latest_trace = None
                return
            match = self.current_match()
            manual_set = self.manual_highlights.get(match.question.number)
            if manual_set:
                self.highlight_blocks = set(manual_set)
                if update_preview:
                    text = self.manual_explanations.get(match.question.number) or self._build_text_from_blocks(manual_set)
                    self.set_preview_text(text)
                    self.preview_status_var.set("Preview (highlight override)")
                return
            try:
                preview, trace = extract_explanation_text_charwise_trace(
                    match,
                    self.index,
                    skip_texts=[],
                    max_mismatches=self.max_mismatches,
                    max_lead=self.max_lead,
                    lookahead_lines=2,
                    stop_on_symbol=True,
                )
            except Exception as exc:
                logging.error("Failed to build preview for Q%s: %s", match.question.number, exc)
                self.preview_status_var.set(f"Preview error: {exc}")
                self.highlight_blocks = set()
                self.latest_auto_preview = ""
                self.latest_trace = None
                return
            self.latest_auto_preview = preview
            self.latest_trace = trace
            highlight: Set[int] = set()
            for rec in trace.get("lines", []):
                if rec.get("kept"):
                    block_idx = rec.get("block_idx")
                    if isinstance(block_idx, int):
                        highlight.add(block_idx)
            self.highlight_blocks = highlight
            if update_preview:
                self.set_preview_text(preview)
                self.preview_status_var.set("Preview updated")
            else:
                self.preview_status_var.set("Preview ready")

        def on_ctrl_press(self, _event: "tk.Event") -> None:
            self.ctrl_pressed = True

        def on_ctrl_release(self, _event: "tk.Event") -> None:
            self.ctrl_pressed = False

        def on_shift_press(self, _event: "tk.Event") -> None:
            self.shift_pressed = True

        def on_shift_release(self, _event: "tk.Event") -> None:
            self.shift_pressed = False

        def drop_current_question(self) -> None:
            if not self.matches:
                messagebox.showinfo("Drop question", "No chunk to drop.")
                return
            match = self.current_match()
            if not messagebox.askyesno(
                "Drop question",
                f"Remove Q{match.question.number} from this session and mark it as missing?",
            ):
                return
            self.missing_questions.append(match.question)
            self.missing_questions.sort(key=lambda q: q.number)
            self.manual_explanations.pop(match.question.number, None)
            self.manual_highlights.pop(match.question.number, None)
            self.matches.pop(self.current_question_idx)
            if self.current_question_idx >= len(self.matches):
                self.current_question_idx = max(0, len(self.matches) - 1)
            self._refresh_question_combo_values()
            self._update_missing_label()
            if self.matches:
                self.load_question(self.current_question_idx)
            else:
                self.load_question(0)

        def on_finish(self) -> None:
            self.finished = True
            self.root.quit()

        def on_cancel(self) -> None:
            if messagebox.askyesno("Cancel review", "Discard chunk edits and exit?"):
                self.cancelled = True
                self.root.quit()

    root = tk.Tk()
    gui = ChunkReviewGUI(root, doc, index, matches, missing_questions, override_path)
    gui.run()
    try:
        root.destroy()
    except Exception:
        pass
    overrides = gui._compute_overrides()
    if gui.cancelled and not gui.finished:
        logging.info("Chunk review cancelled by user.")
        return False, overrides
    logging.info("Chunk review completed.")
    return True, overrides


def launch_omit_gui(doc: fitz.Document, *, zoom: float = 1.0) -> List[OmitRegion]:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        logging.error("Tkinter is not available; cannot launch omit-region GUI.")
        return []

    if doc.page_count == 0:
        logging.warning("PDF has no pages; skipping omit-region GUI.")
        return []

    class OmitGUI:
        def __init__(self, root: "tk.Tk") -> None:
            self.root = root
            self.root.title("Select Header/Footer Regions")
            self.doc = doc
            self.zoom = zoom
            self.page_index = 0
            self.image_cache: Optional["tk.PhotoImage"] = None
            self.canvas = tk.Canvas(self.root, highlightthickness=0)
            self.canvas.grid(row=1, column=0, columnspan=4, sticky="nsew")
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)

            self.instructions = ttk.Label(self.root, text="Drag to draw rectangles to omit. Scroll to zoom. Use Next/Prev to change pages.")
            self.instructions.grid(row=0, column=0, columnspan=3, padx=6, pady=6, sticky="w")

            self.apply_all_var = tk.BooleanVar(value=True)
            self.apply_all_check = ttk.Checkbutton(self.root, text="Apply to all pages", variable=self.apply_all_var)
            self.apply_all_check.grid(row=0, column=3, padx=6, pady=6, sticky="e")

            self.page_label = ttk.Label(self.root, text="")
            self.page_label.grid(row=2, column=0, padx=6, pady=6, sticky="w")

            self.prev_btn = ttk.Button(self.root, text="◀ Prev", command=self.prev_page)
            self.prev_btn.grid(row=2, column=1, padx=6, pady=6, sticky="e")

            self.next_btn = ttk.Button(self.root, text="Next ▶", command=self.next_page)
            self.next_btn.grid(row=2, column=2, padx=6, pady=6, sticky="w")

            self.clear_btn = ttk.Button(self.root, text="Remove Last", command=self.remove_last)
            self.clear_btn.grid(row=2, column=3, padx=6, pady=6, sticky="e")

            self.finish_btn = ttk.Button(self.root, text="Finish", command=self.finish)
            self.finish_btn.grid(row=3, column=2, padx=6, pady=6, sticky="e")

            self.cancel_btn = ttk.Button(self.root, text="Cancel", command=self.cancel)
            self.cancel_btn.grid(row=3, column=1, padx=6, pady=6, sticky="w")

            self.status_var = tk.StringVar(value="")
            self.status_label = ttk.Label(self.root, textvariable=self.status_var)
            self.status_label.grid(row=3, column=0, padx=6, pady=6, sticky="w")

            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
            self.canvas.bind("<MouseWheel>", self.on_scroll)  # Windows / macOS
            self.canvas.bind("<Button-4>", self.on_scroll)    # Linux scroll up
            self.canvas.bind("<Button-5>", self.on_scroll)    # Linux scroll down

            self.start_x: Optional[float] = None
            self.start_y: Optional[float] = None
            self.current_rect_id: Optional[int] = None
            self.cancelled = False
            self.regions: Dict[int, List[fitz.Rect]] = defaultdict(list)
            self.canvas_region_ids: Dict[int, List[int]] = defaultdict(list)
            self.region_actions: List[List[Tuple[int, fitz.Rect]]] = []

            self.load_page()

        def load_page(self) -> None:
            page = self.doc[self.page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), alpha=False)
            data = pix.tobytes("ppm")
            self.image_cache = tk.PhotoImage(data=data)
            self.canvas.delete("all")
            self.canvas.config(width=pix.width, height=pix.height)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_cache)
            ids: List[int] = []
            for rect in self.regions.get(self.page_index, []):
                rect_id = self.canvas.create_rectangle(
                    rect.x0 * self.zoom, rect.y0 * self.zoom, rect.x1 * self.zoom, rect.y1 * self.zoom, outline="#ff6600", width=2
                )
                ids.append(rect_id)
            self.canvas_region_ids[self.page_index] = ids
            self.page_label.config(text=f"Page {self.page_index + 1} / {self.doc.page_count} | Zoom {self.zoom * 100:.0f}%")
            self.update_status()

        def update_status(self) -> None:
            count = len(self.regions.get(self.page_index, []))
            total = sum(len(items) for items in self.regions.values())
            self.status_var.set(f"Page regions: {count} | Total regions: {total}")

        def on_press(self, event: "tk.Event") -> None:
            self.start_x = event.x
            self.start_y = event.y
            self.current_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ff6600", width=2)

        def on_drag(self, event: "tk.Event") -> None:
            if self.current_rect_id is not None and self.start_x is not None:
                self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

        def on_release(self, event: "tk.Event") -> None:
            if self.current_rect_id is None or self.start_x is None or self.start_y is None:
                return
            x0, y0 = self.start_x, self.start_y
            x1, y1 = event.x, event.y
            if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
                self.canvas.delete(self.current_rect_id)
                self.current_rect_id = None
                self.start_x = None
                self.start_y = None
                return
            rect = fitz.Rect(min(x0, x1) / self.zoom, min(y0, y1) / self.zoom, max(x0, x1) / self.zoom, max(y0, y1) / self.zoom)
            rect_id = self.current_rect_id
            pages = list(range(self.doc.page_count)) if self.apply_all_var.get() else [self.page_index]
            stored_rects: List[Tuple[int, fitz.Rect]] = []
            for page_idx in pages:
                rect_copy = fitz.Rect(rect)
                self.regions[page_idx].append(rect_copy)
                stored_rects.append((page_idx, rect_copy))
                if page_idx == self.page_index and rect_id is not None:
                    self.canvas_region_ids[self.page_index].append(rect_id)
            self.region_actions.append(stored_rects)
            self.current_rect_id = None
            self.start_x = None
            self.start_y = None
            self.update_status()

        def remove_last(self) -> None:
            if not self.region_actions:
                return
            stored = self.region_actions.pop()
            for page_idx, rect_obj in stored:
                rect_list = self.regions.get(page_idx)
                if rect_list and rect_obj in rect_list:
                    rect_list.remove(rect_obj)
                    if not rect_list:
                        self.regions.pop(page_idx, None)
            self.load_page()

        def prev_page(self) -> None:
            if self.page_index == 0:
                return
            self.page_index -= 1
            self.load_page()

        def next_page(self) -> None:
            if self.page_index + 1 >= self.doc.page_count:
                return
            self.page_index += 1
            self.load_page()

        def finish(self) -> None:
            self.finished = True
            self.root.quit()

        def cancel(self) -> None:
            if messagebox.askyesno("Cancel", "Discard omit regions and exit?"):
                self.cancelled = True
                self.root.quit()

        def collect(self) -> List[OmitRegion]:
            result: List[OmitRegion] = []
            for page_idx, entries in self.regions.items():
                for rect in entries:
                    result.append(OmitRegion(page_idx, rect))
            return result

        def on_scroll(self, event: "tk.Event") -> None:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = event.delta
            elif hasattr(event, "num"):
                if event.num == 4:
                    delta = 120
                elif event.num == 5:
                    delta = -120
            if delta == 0:
                return
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.5, min(3.0, self.zoom * factor))
            if abs(new_zoom - self.zoom) < 1e-3:
                return
            self.zoom = new_zoom
            self.load_page()

    root = tk.Tk()
    gui = OmitGUI(root)
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    if gui.cancelled:
        logging.info("Omit-region GUI cancelled; proceeding without exclusions.")
        return []
    regions = gui.collect()
    logging.info("Omit-region GUI captured %s regions.", len(regions))
    return regions


# ------------------ CLI & main ------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate question regions in a PDF using metadata from a JSON file."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF to annotate.")
    parser.add_argument("--trace-dir", type=Path,
    help="Directory to dump per-question extraction traces (one JSON per question).")

    parser.add_argument(
        "--json",
        type=Path,
        help="Question metadata JSON (required unless running EasyOCR export only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination PDF path (required when generating annotated output).",
    )
    parser.add_argument("--subject", help="Filter questions by subject.")
    parser.add_argument("--year", type=int, help="Filter questions by exam year.")
    parser.add_argument("--target", help="Filter questions by target/audience.")
    parser.add_argument("--question", type=int, nargs="+", help="Limit to specific question numbers (space-separated).")
    parser.add_argument("--dump-text", type=Path, help="Optional path to write the linearized PDF text stream.")
    parser.add_argument("--padding", type=float, default=4.0, help="Extra padding (points) around detected regions.")
    parser.add_argument("--stroke-width", type=float, default=0.8, help="Rectangle stroke width (points).")
    parser.add_argument("--label-font-size", type=float, default=8.0, help="Label font size (points).")
    parser.add_argument("--label-prefix", default="Q", help="Prefix to place before each question number label.")
    parser.add_argument("--label-offset", type=float, default=6.0, help="Vertical offset (points) between rectangle and label.")
    parser.add_argument("--expand-left", dest="expand_left", action="store_true", default=True, help="Expand table region to left edge.")
    parser.add_argument("--no-expand-left", dest="expand_left", action="store_false", help="Do not expand table region to left edge.")
    parser.add_argument("--expand-right", dest="expand_right", action="store_true", default=True, help="Expand table region to right edge.")
    parser.add_argument("--no-expand-right", dest="expand_right", action="store_false", help="Do not expand table region to right edge.")
    parser.add_argument("--crop-page", dest="crop_page", action="store_true", default=True, help="Crop table region before Camelot.")
    parser.add_argument("--no-crop-page", dest="crop_page", action="store_false", help="Do not crop page before Camelot.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output PDF.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--mismatch-report", type=Path, help="Write JSON diagnostics for unmatched questions.")
    parser.add_argument("--explanation-json", type=Path, help="Write filtered questions with extracted explanations to this JSON file.")
    parser.add_argument("--omit-gui", action="store_true", help="Launch a GUI to select header/footer regions to omit during processing.")
    parser.add_argument("--review-chunks", action="store_true", help="Launch a GUI to review and adjust chunk boundaries before extraction.")
    parser.add_argument(
        "--chunk-overrides",
        type=Path,
        help="Optional JSON file to load/save manual chunk boundary overrides (per question).",
    )
    parser.add_argument("--ocr-export", type=Path, help="Write EasyOCR-derived question chunks to this JSON file.")
    parser.add_argument(
        "--ocr-langs",
        nargs="+",
        dest="ocr_langs",
        help="Language codes for EasyOCR (default: ko en).",
    )
    parser.add_argument("--ocr-gpu", action="store_true", help="Enable GPU acceleration for EasyOCR processing.")
    parser.add_argument("--ocr-dpi", type=int, default=300, help="Rendering DPI for EasyOCR preprocessing.")
    # Charwise tunables
    parser.add_argument("--charwise-max-mismatches", type=int, default=2, help="Max wordlike mismatches while consuming option prefix.")
    parser.add_argument("--charwise-max-lead", type=int, default=6, help="Max leading non-word chars to ignore before option.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not args.pdf.exists():
        logging.error("PDF file not found: %s", args.pdf)
        return 1
    if args.json and not args.json.exists():
        logging.error("JSON file not found: %s", args.json)
        return 1

    questions: List[Question] = []
    if args.json:
        questions = load_questions(
            args.json,
            subject=args.subject,
            year=args.year,
            target=args.target,
            only_numbers=args.question,
        )
        if not questions:
            logging.error("No questions matched the provided filters.")
            return 1
        if args.output is None:
            logging.error("--output is required when processing question metadata.")
            return 1
    elif not args.ocr_export:
        logging.error("Either provide --json or enable --ocr-export for OCR-only extraction.")
        return 1

    if args.output and args.output.exists() and not args.overwrite:
        logging.error("Output file %s already exists (use --overwrite to replace it).", args.output)
        return 1

    with fitz.open(args.pdf) as doc:
        omit_regions: List[OmitRegion] = []
        if args.omit_gui:
            omit_regions = launch_omit_gui(doc)
        index = None
        if questions or args.dump_text or args.review_chunks or args.chunk_overrides or args.mismatch_report or args.explanation_json:
            index = LinearPdfIndex(doc, omit_regions=omit_regions)
            if omit_regions:
                logging.info(
                    "Omitting %s regions across %s pages",
                    len(omit_regions),
                    len({r.page_index for r in omit_regions}),
                )

            if args.dump_text and index:
                logging.info("Writing linearized text to %s", args.dump_text)
                index.dump_text(args.dump_text)

        if args.ocr_export:
            try:
                ocr_payload = perform_easyocr_export(
                    doc,
                    languages=args.ocr_langs,
                    gpu=args.ocr_gpu,
                    dpi=args.ocr_dpi,
                    subject=args.subject,
                    year=args.year,
                    target=args.target,
                )
            except RuntimeError as exc:
                logging.error("Skipping EasyOCR export: %s", exc)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("EasyOCR export failed: %s", exc)
            else:
                destination = args.ocr_export
                parent = destination.parent
                dir_ready = True
                if parent and not parent.exists():
                    try:
                        parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        logging.error("Failed to create directory %s: %s", parent, exc)
                        dir_ready = False
                if dir_ready:
                    try:
                        destination.write_text(
                            json.dumps(ocr_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        logging.info(
                            "Wrote EasyOCR export (%s questions) to %s",
                            len(ocr_payload),
                            args.ocr_export,
                        )
                    except OSError as exc:
                        logging.error("Failed to write EasyOCR export to %s: %s", args.ocr_export, exc)
                if not ocr_payload:
                    logging.warning("EasyOCR export produced no question groups.")

        if args.ocr_export and not questions:
            # Pure OCR extraction; no further chunking work required.
            return 0

        if not index:
            logging.error("Unable to build linear index required for chunking.")
            return 1

        matches, mismatches = match_questions_to_blocks(index, questions)
        baseline_ranges: Dict[int, Tuple[int, int]] = {
            match.question.number: (match.start_block_idx, match.end_block_idx) for match in matches
        }
        matched_numbers = {match.question.number for match in matches}
        missing_questions = [q for q in questions if q.number not in matched_numbers]

        override_map: Dict[int, ChunkOverrideSpec] = {}
        if args.chunk_overrides:
            override_map = load_chunk_overrides(args.chunk_overrides)
            apply_chunk_overrides(matches, override_map, len(index.blocks))
            if override_map:
                logging.info("Applied %s existing chunk overrides.", len(override_map))
        original_override_specs = {
            q: ChunkOverrideSpec(
                start_block_idx=spec.start_block_idx,
                end_block_idx=spec.end_block_idx,
                explanation_override=spec.explanation_override,
            )
            for q, spec in override_map.items()
        }

        if args.review_chunks:
            pre_gui_specs: Dict[int, ChunkOverrideSpec] = {
                match.question.number: ChunkOverrideSpec(
                    match.start_block_idx,
                    match.end_block_idx,
                    match.manual_explanation,
                )
                for match in matches
            }
            completed, gui_overrides = review_chunks_gui(
                doc,
                index,
                matches,
                baseline_ranges=baseline_ranges,
                missing_questions=missing_questions,
                max_mismatches=args.charwise_max_mismatches,
                max_lead=args.charwise_max_lead,
                override_path=args.chunk_overrides,
            )
            if not completed:
                preserved = set(pre_gui_specs.keys())
                matches[:] = [match for match in matches if match.question.number in preserved]
                for match in matches:
                    spec = pre_gui_specs.get(match.question.number)
                    if spec:
                        match.start_block_idx = spec.start_block_idx
                        match.end_block_idx = spec.end_block_idx
                        match.manual_explanation = spec.explanation_override
                override_map = original_override_specs
            else:
                override_map = gui_overrides
                if gui_overrides:
                    logging.info("Captured %s chunk override entries during review.", len(gui_overrides))
                if args.chunk_overrides:
                    save_chunk_overrides(args.chunk_overrides, gui_overrides)

        # Per-page table cache & options
        table_cache: Dict[Tuple[int, bool, bool, bool], List[Dict[str, object]]] = {}
        table_opts = {
            "expand_left": args.expand_left,
            "expand_right": args.expand_right,
            "crop_page": args.crop_page,
            "y_tol": 2.0,
            "gap_space": 1.5,
            "join_lines_with": " ",
        }

        # Reset explanations
        for q in questions:
            content = q.raw_entry.get("content")
            if isinstance(content, dict):
                content.pop("explanation", None)
                content.pop("referenced_table", None)

        explanation_map: Dict[int, str] = {}
        for match in matches:
            referenced_tables = extract_tables_for_match(args.pdf, match, index, table_cache, table_opts)
            content = match.question.raw_entry.setdefault("content", {})
            if referenced_tables:
                sanitized = []
                for table in referenced_tables:
                    table_copy = dict(table)
                    table_copy.pop("bbox",None)
                    sanitized.append(table_copy)
                content["referenced_table"] = sanitized
            skip_texts: List[str] = []
            for tbl in referenced_tables:
                for row in tbl.get("rows", []):
                     for cell in row:
                        if not cell:
                            continue
                        text = str(cell).strip()
                        if not text:
                            continue
                        skip_texts.append(text)
                        skip_texts.extend(part.strip() for part in text.split() if part.strip())
                    #OG at the Bottom
                    #skip_texts.extend(str(cell) for cell in row if cell)

            manual_override = match.manual_explanation if match.manual_explanation else None
            if manual_override:
                explanation_text = manual_override
                trace = {"manual_override": True, "question_number": match.question.number}
            else:
                explanation_text, trace = extract_explanation_text_charwise_trace(
                    match,
                    index,
                    skip_texts=skip_texts,
                    max_mismatches=args.charwise_max_mismatches,
                    max_lead=args.charwise_max_lead,
                    lookahead_lines=2,
                    stop_on_symbol=True,
                )

            explanation_map[match.question.number] = explanation_text
            content["explanation"] = explanation_text

            # Dump per-question trace if requested
            if args.trace_dir:
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                (args.trace_dir / f"q{match.question.number:03}.json").write_text(
                    json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
                )


        logging.info("Extracted explanations for %s questions", len(explanation_map))

        if args.mismatch_report:
            payload = [asdict(item) for item in mismatches]
            try:
                if args.mismatch_report.parent and not args.mismatch_report.parent.exists():
                    args.mismatch_report.parent.mkdir(parents=True, exist_ok=True)
                args.mismatch_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logging.info("Wrote mismatch diagnostics to %s (%s items)", args.mismatch_report, len(payload))
            except OSError as exc:
                logging.error("Failed to write mismatch report to %s: %s", args.mismatch_report, exc)

        if args.explanation_json:
            payload = [q.raw_entry for q in questions]
            try:
                serialized = json.dumps(payload, ensure_ascii=False, indent=2)
                explanation_dest = args.explanation_json
                # When the user passes a directory, drop a PDF-specific JSON file inside it.
                default_name = f"{args.output.stem}.explanations.json"
                if explanation_dest.exists() and explanation_dest.is_dir():
                    explanation_dest = explanation_dest / default_name
                elif not explanation_dest.exists() and not explanation_dest.suffix:
                    explanation_dest.mkdir(parents=True, exist_ok=True)
                    explanation_dest = explanation_dest / default_name
                parent = explanation_dest.parent
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                explanation_dest.write_text(serialized, encoding="utf-8")
                logging.info("Wrote explanations to %s", explanation_dest)
            except OSError as exc:
                logging.error("Failed to write explanation JSON to %s: %s", args.explanation_json, exc)
            except (TypeError, ValueError) as exc:
                logging.error("Failed to serialize explanation JSON: %s", exc)

        if not matches:
            logging.error("No questions matched textual content in the PDF.")
            return 2

        annotate_pdf(
            doc,
            index,
            matches,
            padding=args.padding,
            stroke_width=args.stroke_width,
            label_font_size=args.label_font_size,
            label_prefix=args.label_prefix,
            text_offset=args.label_offset,
        )

        logging.info("Saving annotated PDF to %s", args.output)
        doc.save(args.output, garbage=4, deflate=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
