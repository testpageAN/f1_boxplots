#!/usr/bin/env python3
"""
F1 Race Lap Analysis
Διαβάζει γυρολόγιο από επίσημο PDF F1 Race Lap Analysis και δημιουργεί box plot.
"""

from pathlib import Path
import re
from collections import defaultdict

import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt

from my_test import extract_race

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
PDF_PATH = "/home/alexis/Documents/python_lessons/formula_1/data_files/2026_03_jpn_f1_r0_timing_racelapanalysis_v01.pdf"

# Extract country
country = extract_race(PDF_PATH)

# Επιθέτα οδηγών με κεφαλαία (όπως εμφανίζονται στο PDF)
SELECTED_DRIVERS = ["ANTONELLI", "PIASTRI", "LECLERC", "RUSSELL", "NORRIS", "HAMILTON"]
OTHER_DRIVERS = ["VERSTAPPEN"]

charts_directory = Path(__file__).parent / "charts_plots"
charts_directory.mkdir(exist_ok=True)
OUTPUT_IMAGE = charts_directory / f"lap_times_boxplot_{country}.png"

# Εύρος Y-άξονα (M:SS.mmm). None = αυτόματο εύρος (εμφανίζονται όλα).
Y_MIN = "1:32.000"
Y_MAX = "1:37.000"
# Y_MIN = None
# Y_MAX = None
# ──────────────────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"^P?(\d+):(\d{2}\.\d{3})$")
_START_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def to_seconds(token: str) -> float | None:
    """
    Μετατρέπει χρόνο γύρου σε δευτερόλεπτα.
    - Αποδέχεται  M:SS.mmm  ή  PM:SS.mmm  (αφαιρεί το P).
    - Επιστρέφει None για την ώρα εκκίνησης LAP 1 (HH:MM:SS).
    """
    t = token.strip()
    if _START_RE.match(t):          # π.χ. 15:05:02  → ώρα εκκίνησης, αγνοείται
        return None
    m = _TIME_RE.match(t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


# ── PDF parsing helpers ────────────────────────────────────────────────────────

def _find_driver_headers(rows: dict[int, list]) -> list[dict]:
    """
    Ανιχνεύει τις επικεφαλίδες οδηγών στη σελίδα.
    Μοτίβο: <1-2ψήφιος αριθμός>  <Όνομα>  <ΕΠΙΘΕΤΟ>
    π.χ. "1  Lando  NORRIS"
    Επιστρέφει list[dict] με: name, x_start (x αριθμού αμαξιού), header_y.
    """
    headers = []
    for y, rw in sorted(rows.items()):
        rw = sorted(rw, key=lambda w: w["x0"])
        texts = [w["text"] for w in rw]
        i = 0
        while i < len(texts) - 2:
            if (
                re.fullmatch(r"\d{1,2}", texts[i])           # αριθμός αμαξιού
                and texts[i + 1][0].isupper()
                and not texts[i + 1].isupper()               # TitleCase όνομα
                and texts[i + 2].isupper()
                and len(texts[i + 2]) > 2                    # ALL_CAPS επίθετο
            ):
                headers.append(
                    {
                        "name": texts[i + 2],
                        "x_start": rw[i]["x0"],
                        "header_y": y,
                    }
                )
                i += 3
            else:
                i += 1
    return headers


def _assign_section_bounds(headers: list[dict], page_width: float) -> None:
    """
    Υπολογίζει x_min / x_max για κάθε τμήμα οδηγού.
    Ομαδοποιεί πρώτα επικεφαλίδες που βρίσκονται στην ίδια οριζόντια σειρά
    (κοντινό header_y), ώστε οδηγοί σε 2η σειρά σελίδας (π.χ. DNF οδηγός
    κάτω από κύρια 3-στήλη διάταξη) να μην διαταράσσουν τα x-όρια της 1ης.
    """
    if not headers:
        return

    # Ταξινόμηση ανά header_y και x_start
    headers.sort(key=lambda h: (h["header_y"], h["x_start"]))

    # Ομαδοποίηση ανά οριζόντια σειρά (y-tolerance = 15 pt)
    y_tol = 15
    groups: list[list[dict]] = []
    group: list[dict] = []
    for hdr in headers:
        if group and abs(hdr["header_y"] - group[0]["header_y"]) > y_tol:
            groups.append(group)
            group = []
        group.append(hdr)
    if group:
        groups.append(group)

    # Ανάθεση x ορίων εντός κάθε ομάδας
    for grp in groups:
        grp.sort(key=lambda w: w["x_start"])
        for k, hdr in enumerate(grp):
            hdr["x_min"] = 0.0 if k == 0 else hdr["x_start"]
            hdr["x_max"] = (
                grp[k + 1]["x_start"] if k < len(grp) - 1 else page_width
            )


def _parse_row(tokens: list[str]) -> list[tuple[int, float]]:
    """
    Αναλύει τα tokens μιας γραμμής PDF σε ζεύγη (αριθμός γύρου, δευτερόλεπτα).
    Μια γραμμή μπορεί να έχει 1 ή 2 εγγραφές (αριστερή/δεξιά υποστήλη).
    Παραδείγματα tokens:
      ['2',  '1:27.344']
      ['11', 'P', '1:43.391']
      ['2',  '1:27.344', '30', '1:23.961']
      ['11', 'P', '1:43.391', '41', 'P', '1:38.866']
    """
    pairs = []
    i = 0
    while i < len(tokens):
        m = re.fullmatch(r"(\d+)P?", tokens[i])   # αριθμός γύρου (± trailing P)
        if m:
            lap_no = int(m.group(1))
            i += 1
            if i < len(tokens) and tokens[i] == "P":
                i += 1                             # κατανάλωσε standalone P
            if i < len(tokens):
                secs = to_seconds(tokens[i])
                if secs is not None and lap_no > 1:   # παράλειψη LAP 1
                    pairs.append((lap_no, secs))
                i += 1                             # κατανάλωσε χρόνο
        else:
            i += 1
    return pairs


# ── Main extraction ────────────────────────────────────────────────────────────

def extract_laps(
    pdf_path: str, wanted: list[str]
) -> dict[str, list[tuple[int, float]]]:
    """
    Ανοίγει το PDF και επιστρέφει dict:
      { 'NORRIS': [(2, 87.34), (3, 86.86), ...], ... }
    """
    wanted_up = {w.upper() for w in wanted}
    result: dict[str, list[tuple[int, float]]] = {w: [] for w in wanted_up}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=4, y_tolerance=4)
            if not words:
                continue

            # Ομαδοποίηση λέξεων ανά γραμμή (στρογγυλοποίηση y)
            rows: dict[int, list] = defaultdict(list)
            for w in words:
                rows[round(w["top"])].append(w)

            # Εύρεση επικεφαλίδων οδηγών στη σελίδα
            all_headers = _find_driver_headers(rows)
            if not all_headers:
                continue
            _assign_section_bounds(all_headers, float(page.width))

            for h in all_headers:
                if h["name"] not in wanted_up:
                    continue

                name = h["name"]
                x_min, x_max = h["x_min"], h["x_max"]
                header_y = h["header_y"]

                # Λέξεις εντός του τμήματος αυτού οδηγού, κάτω από την επικεφαλίδα
                sec_rows: dict[int, list] = defaultdict(list)
                for w in words:
                    if x_min <= w["x0"] < x_max and round(w["top"]) > header_y + 5:
                        sec_rows[round(w["top"])].append(w)

                for y in sorted(sec_rows):
                    rw = sorted(sec_rows[y], key=lambda w: w["x0"])
                    texts = [w["text"] for w in rw]
                    if "LAP" in texts or "TIME" in texts:
                        continue                        # παράλειψη επικεφαλίδων στηλών
                    for lap_no, secs in _parse_row(texts):
                        result[name].append((lap_no, secs))

    return result


# ── DataFrame ─────────────────────────────────────────────────────────────────

def build_dataframe(laps: dict[str, list[tuple[int, float]]]) -> pd.DataFrame:
    records = [
        {"Driver": driver, "Lap": lap, "LapTime_s": secs}
        for driver, lap_list in laps.items()
        for lap, secs in sorted(lap_list, key=lambda x: x[0])
    ]
    return pd.DataFrame(records, columns=["Driver", "Lap", "LapTime_s"])


# ── Box plot ──────────────────────────────────────────────────────────────────

def plot_boxplot(df: pd.DataFrame, drivers: list[str], output: str) -> None:
    drivers_ok = [d.upper() for d in drivers if d.upper() in df["Driver"].unique()]
    if not drivers_ok:
        print("Δεν βρέθηκαν δεδομένα για τους επιλεγμένους οδηγούς.")
        return

    data = [df.loc[df["Driver"] == d, "LapTime_s"].values for d in drivers_ok]

    fig, ax = plt.subplots(figsize=(max(8, len(drivers_ok) * 2.5), 6))

    # Αν έχει οριστεί εύρος Y, κρύβουμε τα outliers (εκτός κλίμακας)
    hide_fliers = Y_MIN is not None or Y_MAX is not None

    bp = ax.boxplot(
        data,
        tick_labels=drivers_ok,
        patch_artist=True,
        notch=False,
        showfliers=not hide_fliers,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5, linestyle="none"),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Χρώματα για κάθε οδηγό
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
               "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Μορφοποίηση άξονα Y σε M:SS.mmm
    def fmt_time(x, _):
        if x < 0:
            return ""
        mins, secs = divmod(x, 60)
        return f"{int(mins)}:{secs:06.3f}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_time))
    ax.set_ylabel("Χρόνος γύρου")
    ax.set_xlabel("Οδηγός")
    ax.set_title("F1 Australian GP 2026 – Κατανομή Χρόνων Γύρου")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Εφαρμογή εύρους Y-άξονα
    if Y_MIN is not None or Y_MAX is not None:
        y_lo = to_seconds(Y_MIN) if Y_MIN else None
        y_hi = to_seconds(Y_MAX) if Y_MAX else None
        ax.set_ylim(bottom=y_lo, top=y_hi)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nΔιάγραμμα αποθηκεύτηκε → {output}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PDF: {PDF_PATH}")
    print(f"Οδηγοί: {SELECTED_DRIVERS}\n")

    laps = extract_laps(PDF_PATH, SELECTED_DRIVERS)

    for driver, lap_list in laps.items():
        print(f"  {driver}: {len(lap_list)} γύροι")

    df = build_dataframe(laps)

    print("\nDataFrame (πρώτες 10 γραμμές):")
    print(df.head(10).to_string(index=False))
    print(f"\nΣύνολο γραμμών: {len(df)}")

    plot_boxplot(df, SELECTED_DRIVERS, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
