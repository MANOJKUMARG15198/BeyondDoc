"""Tokenizer evaluation on the Batch 1 benchmark sample (P2-1 is pending: nothing is chosen here).

Measures how candidate tokenizers count the project's medical text, so the project owner can choose the tokenizer that
enforces the chunk ceiling. Text extraction is a light, read-only approximation of the future parsers
(section text only; package panels, product data, Highlights and plain reference lists left out, as
approved). Packing mimics chunking only far enough to compare token counts; it is not the chunker.

.venv/Scripts/python -X utf8 scripts/phase2/evaluate_tokenizers.py
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import io
import re
import statistics
import sys
import tarfile
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from tokenizers import BertWordPieceTokenizer, Tokenizer

MANIFESTS = Path("data/manifests/batches")
OUT = Path("data/processed/chunking/reports")
BERT_FILE = Path(
    "data/models/tokenizers/google-bert__bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json"
)
BIOMED_FILE = Path(
    "data/models/tokenizers/microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/"
    "e1354b7a3a09615f6aba48dfad4b7a613eef7062/vocab.txt"
)
CEILING = 510  # 512 model positions minus [CLS] and [SEP]
V = "{urn:hl7-org:v3}"
SPL_EXCLUDED = {"51945-4": "package_label", "48780-1": "product_data"}
LIVERTOX_SKIPPED = ("OTHER_REFERENCE_LINKS", "CITED_REFERENCE", "CITED_REFERENCES")

# The 20 documents of DATA_AUDIT_BATCH_01.md section 8.
SAMPLE = {
    "dailymed": [
        "1e11ad30-1041-4520-10b0-8f9d30d30fcc",  # Synthroid
        "290a91e5-22ae-4d62-ab34-39a3117002e2",  # Prednisone
        "a60cc18b-0631-4cf0-b021-9f52224ece65",  # Lipitor
        "d5228e81-9c86-4781-b984-20feaa2ae779",  # Metformin
        "c040bd1d-45b7-49f2-93ea-aed7220b30ac",  # Abilify
        "40a68a27-4422-5ba0-e063-6294a90ac550",  # Oxycodone/APAP
        "5a852412-69af-6502-e063-6294a90a0748",  # Diclofenac
        "5a7a1141-d3b1-0ee8-e063-6294a90a16aa",  # SurviveX
        "660ac9df-f1b1-4c89-94dd-9fae0a013f3c",  # Claritin
        "dd379cdd-90ab-42e0-ad89-f50d3220f611",  # Mucinex
    ],
    "fda_drugsatfda": ["NDA021402:27674", "NDA020702:78041", "NDA018936:78425"],
    "pmc_oa": ["PMC13004290.1", "PMC3787571.1", "PMC13509944.1", "PMC13529079.1"],
    "nih_livertox": ["Cyclobenzaprine", "Gabapentin", "Atorvastatin"],
}
SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\[])")
WORD = re.compile(r"[A-Za-z][A-Za-z-]{5,}")


@dataclass(frozen=True, slots=True)
class Unit:
    document_id: str
    source: str
    section: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split()))


def checked(path: str, sha256: str) -> bytes:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"checksum mismatch: {path}")
    return data


# ---- light extraction (read-only, in memory) ----


def spl_units(document_id: str, xml: bytes) -> list[Unit]:
    units: list[Unit] = []

    def walk(sec: ET.Element, path: str) -> None:
        code_el = sec.find(V + "code")
        if code_el is not None and code_el.get("code") in SPL_EXCLUDED:
            return
        title_el = sec.find(V + "title")
        title = norm("".join(title_el.itertext())) if title_el is not None else ""
        paras = [title] if title else []
        text_el = sec.find(V + "text")  # <excerpt> (Highlights) is a sibling of <text>, so it is left out
        if text_el is not None:
            if text_el.text and text_el.text.strip():
                paras.append(norm(text_el.text))
            paras += [p for p in (norm("".join(c.itertext())) for c in text_el) if p]
        if len(paras) > (1 if title else 0):
            units.append(Unit(document_id, "dailymed", path, tuple(paras)))
        for i, comp in enumerate(sec.findall(V + "component"), 1):
            child = comp.find(V + "section")
            if child is not None:
                walk(child, f"{path}.{i}")

    body = ET.fromstring(xml).find(f".//{V}structuredBody")
    for i, comp in enumerate(body.findall(V + "component"), 1):
        sec = comp.find(V + "section")
        if sec is not None:
            walk(sec, str(i))
    return units


def jats_units(document_id: str, source: str, root: ET.Element, skipped: tuple[str, ...] = ()) -> list[Unit]:
    units: list[Unit] = []
    for i, abstract in enumerate(root.iter("abstract"), 1):
        paras = tuple(p for p in (norm("".join(c.itertext())) for c in abstract) if p)
        if paras:
            units.append(Unit(document_id, source, f"abstract{i}", paras))

    def walk(sec: ET.Element, path: str) -> None:
        if sec.get("id", "").endswith(skipped):
            return
        title_el = sec.find("title")
        paras = [norm("".join(title_el.itertext()))] if title_el is not None else []
        for child in sec:
            if child.tag in ("title", "sec"):
                continue
            if child.tag == "ref-list":  # LiverTox annotated bibliography: one paragraph per reference
                paras += [norm("".join(r.itertext())) for r in child.iter("ref")]
            else:
                paras.append(norm("".join(child.itertext())))
        paras = [p for p in paras if p]
        if paras:
            units.append(Unit(document_id, source, path, tuple(paras)))
        for i, child in enumerate(sec.findall("sec"), 1):
            walk(child, f"{path}.{i}")

    body = root.find(".//body")  # <back> (PMC reference list) is outside <body>, so it is left out
    for i, sec in enumerate(body.findall("sec"), 1):
        walk(sec, str(i))
    return units


def pdf_unit(document_id: str, data: bytes) -> Unit:
    lines = [
        norm(line) for page in PdfReader(io.BytesIO(data)).pages for line in (page.extract_text() or "").splitlines()
    ]
    return Unit(document_id, "fda_drugsatfda", "letter", tuple(line for line in lines if line))


def load_sample() -> list[Unit]:
    rows = list(csv.DictReader((MANIFESTS / "batch_01_downloaded.csv").open(encoding="utf-8")))
    units: list[Unit] = []
    for r in rows:
        sid, fmt = r["source_document_id"], r["format"]
        if r["source_name"] == "dailymed" and fmt == "zip" and sid in SAMPLE["dailymed"]:
            with zipfile.ZipFile(io.BytesIO(checked(r["file"], r["sha256"]))) as zf:
                name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
                units += spl_units(r["logical_document_id"], zf.read(name))
        elif r["source_name"] == "pmc_oa" and fmt == "xml" and sid in SAMPLE["pmc_oa"]:
            root = ET.fromstring(checked(r["file"], r["sha256"]))
            units += jats_units(r["logical_document_id"], "pmc_oa", root)
        elif r["source_name"] == "fda_drugsatfda" and sid in SAMPLE["fda_drugsatfda"]:
            units.append(pdf_unit(r["logical_document_id"], checked(r["file"], r["sha256"])))
    members = list(csv.DictReader((MANIFESTS / "batch_01_livertox_members.csv").open(encoding="utf-8")))
    archive = next(r for r in rows if r["source_name"] == "nih_livertox")
    with tarfile.open(fileobj=io.BytesIO(checked(archive["file"], archive["sha256"])), mode="r:gz") as tf:
        for m in members:
            name = Path(m["member_nxml"]).stem
            if name in SAMPLE["nih_livertox"]:
                root = ET.fromstring(tf.extractfile(m["member_nxml"]).read())
                units += jats_units(f"nih_livertox:{m['source_document_id']}", "nih_livertox", root, LIVERTOX_SKIPPED)
    return units


# ---- tokenizers and packing ----


def load_tokenizers() -> dict[str, Tokenizer | BertWordPieceTokenizer]:
    return {
        "bert_uncased": Tokenizer.from_file(str(BERT_FILE)),
        "biomedbert": BertWordPieceTokenizer(str(BIOMED_FILE), lowercase=True),
    }


def counter(tok: Tokenizer | BertWordPieceTokenizer) -> Callable[[str], int]:
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def fit_pieces(text: str, count: Callable[[str], int], limit: int) -> list[str]:
    """Split one paragraph so every piece fits: sentences first, then words."""
    if count(text) <= limit:
        return [text]
    sentences = SENTENCE_BREAK.split(text)
    if len(sentences) > 1:
        return [p for s in sentences for p in fit_pieces(s, count, limit)]
    pieces, current = [], []
    for word in text.split():
        if current and count(" ".join([*current, word])) > limit:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    return [*pieces, " ".join(current)]


def pack(unit: Unit, count: Callable[[str], int], limit: int = CEILING) -> list[str]:
    windows, current = [], []
    for para in unit.paragraphs:
        for piece in fit_pieces(para, count, limit):
            if current and count("\n".join([*current, piece])) > limit:
                windows.append("\n".join(current))
                current = []
            current.append(piece)
    return [*windows, "\n".join(current)] if current else windows


def peak_memory_mb() -> float:
    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t)
            for n in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return float("nan")
    return c.PeakWorkingSetSize / 2**20


def pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


# ---- report ----


def main() -> int:
    units = load_sample()
    toks = load_tokenizers()
    counts = {name: counter(t) for name, t in toks.items()}
    words = {u: len(u.text.split()) for u in units}
    tokens = {name: {u: count(u.text) for u in units} for name, count in counts.items()}
    lines = [
        "# Tokenizer evaluation — Batch 1 benchmark sample",
        "",
        f"{len({u.document_id for u in units})} documents · {len(units)} sections · {sum(words.values()):,} words · "
        f"ceiling {CEILING} content tokens. Light extraction, not the parser. **No tokenizer is chosen here (P2-1).**",
        "",
        "## 1. Token density (tokens per word)",
        "",
        "| Source | Words | " + " | ".join(f"{n} tokens | {n} tok/word" for n in toks) + " |",
        "|---|---:|" + "---:|---:|" * len(toks),
    ]
    for source in [*SAMPLE, "all"]:
        us = [u for u in units if source in ("all", u.source)]
        w = sum(words[u] for u in us)
        cells = " | ".join(f"{sum(tokens[n][u] for u in us):,} | {sum(tokens[n][u] for u in us) / w:.2f}" for n in toks)
        lines.append(f"| {source} | {w:,} | {cells} |")

    vocab = sorted({m.group().lower() for u in units for m in WORD.finditer(u.text)})
    pieces = {n: {v: counts[n](v) for v in vocab} for n in toks}
    lines += [
        "",
        "## 2. Fragmentation of words of 6+ letters",
        "",
        f"{len(vocab):,} distinct words.",
        "",
        "| Tokenizer | 1 piece | 2 pieces | ≥3 pieces | mean pieces |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in toks:
        p = list(pieces[n].values())
        lines.append(
            f"| {n} | {sum(x == 1 for x in p) / len(p):.1%} | {sum(x == 2 for x in p) / len(p):.1%} | "
            f"{sum(x >= 3 for x in p) / len(p):.1%} | {statistics.mean(p):.2f} |"
        )
    plan = list(csv.DictReader((MANIFESTS / "batch_01_plan.csv").open(encoding="utf-8")))
    drugs = sorted({w for r in plan for w in re.findall(r"[a-z]{5,}", r["drug"].lower())})
    worst = sorted(drugs, key=lambda d: -counts["bert_uncased"](d))[:25]
    lines += [
        "",
        "Drug names from the Batch 1 plan (25 most fragmented in bert_uncased):",
        "",
        "| Drug | " + " | ".join(toks) + " | bert_uncased pieces |",
        "|---|" + "---:|" * len(toks) + "---|",
    ]
    for d in worst:
        lines.append(
            f"| {d} | " + " | ".join(str(counts[n](d)) for n in toks) + " | "
            f"`{' '.join(toks['bert_uncased'].encode(d, add_special_tokens=False).tokens)}` |"
        )

    chars = sorted({c for u in units for c in u.text if ord(c) > 127})
    lines += ["", "## 3. Unknown tokens ([UNK]: information lost to the model, not to stored text)", ""]
    for n, t in toks.items():
        unk = [c for c in chars if "[UNK]" in t.encode(c, add_special_tokens=False).tokens]
        total = sum(t.encode(u.text, add_special_tokens=False).tokens.count("[UNK]") for u in units)
        lines.append(f"- **{n}:** {total:,} [UNK] tokens; characters mapped to [UNK]: {' '.join(unk) or 'none'}")

    lines += [
        "",
        f"## 4. Cross-check: windows packed to ≤{CEILING} by one tokenizer, measured by the other",
        "",
        "| Packed by | Windows | median | P95 | max | measured by | over ceiling | max |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for a in toks:
        wins = [w for u in units for w in pack(u, counts[a])]
        own = [counts[a](w) for w in wins]
        for b in (x for x in toks if x != a):
            other = [counts[b](w) for w in wins]
            over = sum(x > CEILING for x in other)
            lines.append(
                f"| {a} | {len(wins):,} | {statistics.median(own):.0f} | {pct(own, 0.95)} | {max(own)} | {b} | "
                f"{over} ({over / len(wins):.1%}) | {max(other)} |"
            )

    factor = statistics.median(tokens["bert_uncased"][u] / words[u] for u in units if words[u] >= 20)

    def estimate(text: str) -> int:
        return round(len(text.split()) * factor)

    est_wins = [w for u in units for w in pack(u, estimate)]
    real = [counts["bert_uncased"](w) for w in est_wins]
    over = sum(x > CEILING for x in real)
    need = max(counts["bert_uncased"](w) / len(w.split()) for w in est_wins if w.split())
    lines += [
        "",
        "## 5. Word-count estimate (words × median factor, fitted on this sample)",
        "",
        f"- Factor: {factor:.2f} bert_uncased tokens per word (in-sample, so optimistic).",
        f"- Windows the estimate packs to ≤{CEILING}: {len(est_wins):,}; really over the ceiling in bert_uncased: "
        f"**{over} ({over / len(est_wins):.1%})**, max {max(real)}.",
        f"- Factor needed for zero overshoot on this sample: {need:.2f} tokens/word.",
    ]

    lines += [
        "",
        "## 6. Speed and memory (this machine, CPU)",
        "",
        "| Tokenizer | tokens/s (best of 3) |",
        "|---|---:|",
    ]
    texts = [u.text for u in units]
    for n, t in toks.items():
        best = min(_timed(t, texts) for _ in range(3))
        lines.append(f"| {n} | {sum(tokens[n].values()) / best:,.0f} |")
    lines += [
        "",
        f"Peak process memory for the whole run: {peak_memory_mb():.0f} MB.",
        "",
        "## Inputs",
        "",
        f"- bert_uncased: `{BERT_FILE}` sha256 `{hashlib.sha256(BERT_FILE.read_bytes()).hexdigest()}`",
        f"- biomedbert: `{BIOMED_FILE}` sha256 `{hashlib.sha256(BIOMED_FILE.read_bytes()).hexdigest()}`",
        "- Raw files: SHA-256 verified against data/manifests/batches/batch_01_downloaded.csv; read in memory only.",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokenizer_eval_batch_01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "tokenizer_eval_batch_01.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["document_id", "source", "section", "words", *toks])
        for u in units:
            writer.writerow([u.document_id, u.source, u.section, words[u], *(tokens[n][u] for n in toks)])
    print("\n".join(lines))
    return 0


def _timed(tok: Tokenizer | BertWordPieceTokenizer, texts: list[str]) -> float:
    start = time.perf_counter()
    tok.encode_batch(texts, add_special_tokens=False)
    return time.perf_counter() - start


if __name__ == "__main__":
    sys.exit(main())
