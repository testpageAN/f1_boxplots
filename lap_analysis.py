#!/usr/bin/env python3
"""
F1 Race Lap Analysis
Διαβάζει γυρολόγιο από επίσημο PDF F1 Race Lap Analysis και δημιουργεί box plot.

Ροή εκτέλεσης:
  1. Ανοίγει το PDF με pdfplumber και εντοπίζει τις στήλες κάθε οδηγού.
  2. Εξάγει τους χρόνους γύρου (σε δευτερόλεπτα) για τους επιλεγμένους οδηγούς.
  3. Φτιάχνει DataFrame με pandas.
  4. Δημιουργεί box plot με matplotlib και αποθηκεύει ως PNG.
"""

from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt

from my_test import extract_race

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Όνομα αρχείου PDF (πρέπει να βρίσκεται στον φάκελο data_files/)
PDF_FILENAME = "2026_02_chn_f1_r0_timing_racelapanalysis_v01.pdf"
PDF_PATH = Path(__file__).parent / "data_files" / PDF_FILENAME

# Εξαγωγή ονόματος χώρας από το όνομα του αρχείου (π.χ. "chn" → "China")
country = extract_race(PDF_FILENAME)

# Οδηγοί για τους οποίους θα δημιουργηθεί το box plot.
# Τα επίθετα πρέπει να είναι κεφαλαία, όπως εμφανίζονται στο PDF.
SELECTED_DRIVERS = ["ANTONELLI", "PIASTRI", "LECLERC", "RUSSELL", "NORRIS", "HAMILTON"]
# Οδηγοί που υπάρχουν στο PDF αλλά δεν συμπεριλαμβάνονται στο γράφημα.
OTHER_DRIVERS = ["VERSTAPPEN"]

# Φάκελος αποθήκευσης γραφημάτων (δημιουργείται αυτόματα αν δεν υπάρχει)
charts_directory = Path(__file__).parent / "charts_plots"
charts_directory.mkdir(exist_ok=True)
OUTPUT_IMAGE = charts_directory / f"lap_times_boxplot_{country}.png"

# Εύρος Y-άξονα σε μορφή "M:SS.mmm".
# Αν και οι δύο είναι None → αυτόματος υπολογισμός βάσει whiskers (χωρίς outliers):
#   Y_MIN = χαμηλότερο whisker όλων των οδηγών − 1 δευτερόλεπτο
#   Y_MAX = υψηλότερο whisker όλων των οδηγών + 1 δευτερόλεπτο
# Παράδειγμα χειροκίνητου ορισμού:
# Y_MIN = "1:32.000"
# Y_MAX = "1:37.000"
Y_MIN = None
Y_MAX = None
# ──────────────────────────────────────────────────────────────────────────────

# Regex για χρόνο γύρου: προαιρετικό "P" (pit lap) + M:SS.mmm
# Παραδείγματα αποδεκτών τιμών: "1:27.344", "P1:43.391"
_TIME_RE = re.compile(r"^P?(\d+):(\d{2}\.\d{3})$")

# Regex για ώρα εκκίνησης αγώνα (HH:MM:SS), π.χ. "15:05:02" — αγνοείται
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

    Το PDF διατάσσει κάθε οδηγό σε στήλη με επικεφαλίδα της μορφής:
      <αριθμός αμαξιού>  <Όνομα>  <ΕΠΙΘΕΤΟ>
    π.χ.  "4  Lando  NORRIS"

    Κριτήρια αναγνώρισης:
      - 1ο token: 1–2ψήφιος αριθμός (αριθμός αμαξιού)
      - 2ο token: TitleCase (μικρό όνομα οδηγού)
      - 3ο token: ALL CAPS με >2 χαρακτήρες (επίθετο οδηγού)

    Επιστρέφει list[dict] με πεδία:
      name      : επίθετο οδηγού (ALL CAPS)
      x_start   : οριζόντια θέση αρχής της στήλης
      header_y  : κατακόρυφη θέση της επικεφαλίδας
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
    Υπολογίζει τα οριζόντια όρια (x_min, x_max) κάθε στήλης οδηγού.

    Το PDF μπορεί να έχει 2 ή 3 οδηγούς ανά σελίδα σε οριζόντια σειρά.
    Κάποιοι οδηγοί (π.χ. DNF) εμφανίζονται σε 2η σειρά κάτω από τους κύριους.
    Η ομαδοποίηση ανά οριζόντια σειρά (y-tolerance 15 pt) εμποδίζει οδηγούς
    της 2ης σειράς να παρεμβαίνουν στα x-όρια της 1ης.

    Κάθε επικεφαλίδα αποκτά:
      x_min : αριστερό όριο στήλης (= x_start του αμέσως επόμενου οδηγού,
               ή 0 για τον πρώτο οδηγό κάθε σειράς)
      x_max : δεξί όριο στήλης (= x_start του επόμενου οδηγού,
               ή page_width για τον τελευταίο)
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
            # Ο πρώτος οδηγός ξεκινά από την άκρη της σελίδας (x=0)
            hdr["x_min"] = 0.0 if k == 0 else hdr["x_start"]
            # Το δεξί όριο είναι το x_start του επόμενου, ή το πλάτος σελίδας
            hdr["x_max"] = (
                grp[k + 1]["x_start"] if k < len(grp) - 1 else page_width
            )


def _parse_row(tokens: list[str]) -> list[tuple[int, float]]:
    """
    Αναλύει τα tokens μιας γραμμής PDF σε ζεύγη (αριθμός γύρου, δευτερόλεπτα).

    Κάθε γραμμή στήλης οδηγού μπορεί να περιέχει 1 ή 2 εγγραφές
    (αριστερή και δεξιά υποστήλη). Το "P" υποδηλώνει pit lap.

    Παραδείγματα tokens:
      ['2',  '1:27.344']                        → 1 εγγραφή
      ['11', 'P', '1:43.391']                   → 1 εγγραφή (pit)
      ['2',  '1:27.344', '30', '1:23.961']      → 2 εγγραφές
      ['11', 'P', '1:43.391', '41', 'P', '1:38.866']  → 2 εγγραφές (pit)

    Ο LAP 1 παραλείπεται γιατί περιέχει ώρα εκκίνησης (HH:MM:SS), όχι χρόνο γύρου.
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
    Ανοίγει το PDF και εξάγει τους χρόνους γύρου για τους επιλεγμένους οδηγούς.

    Επιστρέφει dict της μορφής:
      { 'NORRIS': [(2, 87.34), (3, 86.86), ...], 'LECLERC': [...], ... }

    Κάθε tuple περιέχει (αριθμός_γύρου, χρόνος_σε_δευτερόλεπτα).
    """
    wanted_up = {w.upper() for w in wanted}
    result: dict[str, list[tuple[int, float]]] = {w: [] for w in wanted_up}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=4, y_tolerance=4)
            if not words:
                continue

            # Ομαδοποίηση λέξεων ανά γραμμή: στρογγυλοποιούμε το y για να
            # συγχωνεύσουμε λέξεις που ανήκουν στην ίδια οριζόντια γραμμή.
            rows: dict[int, list] = defaultdict(list)
            for w in words:
                rows[round(w["top"])].append(w)

            # Εύρεση επικεφαλίδων οδηγών στη σελίδα και υπολογισμός x-ορίων
            all_headers = _find_driver_headers(rows)
            if not all_headers:
                continue
            _assign_section_bounds(all_headers, float(page.width))

            # Για κάθε οδηγό που μας ενδιαφέρει, συλλέγουμε τους χρόνους γύρου
            for h in all_headers:
                if h["name"] not in wanted_up:
                    continue

                name = h["name"]
                x_min, x_max = h["x_min"], h["x_max"]
                header_y = h["header_y"]

                # Κρατάμε μόνο τις λέξεις που βρίσκονται:
                #   - οριζόντια: εντός των ορίων της στήλης του οδηγού
                #   - κατακόρυφα: κάτω από την επικεφαλίδα (+ 5 pt περιθώριο)
                sec_rows: dict[int, list] = defaultdict(list)
                for w in words:
                    if x_min <= w["x0"] < x_max and round(w["top"]) > header_y + 5:
                        sec_rows[round(w["top"])].append(w)

                # Ανάλυση κάθε γραμμής της στήλης
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
    """
    Μετατρέπει το dict με τους χρόνους γύρου σε pandas DataFrame.

    Στήλες: Driver (str), Lap (int), LapTime_s (float).
    Οι γύροι κάθε οδηγού ταξινομούνται αύξοντα κατά αριθμό γύρου.
    """
    records = [
        {"Driver": driver, "Lap": lap, "LapTime_s": secs}
        for driver, lap_list in laps.items()
        for lap, secs in sorted(lap_list, key=lambda x: x[0])
    ]
    return pd.DataFrame(records, columns=["Driver", "Lap", "LapTime_s"])


# ── Box plot ──────────────────────────────────────────────────────────────────

def plot_boxplot(df: pd.DataFrame, drivers: list[str], output: str) -> None:
    """
    Δημιουργεί box plot με τους χρόνους γύρου των επιλεγμένων οδηγών.

    Τα outliers δεν εμφανίζονται ποτέ στο γράφημα.

    Εύρος Y-άξονα:
      - Αν Y_MIN / Y_MAX είναι ορισμένα: χρησιμοποιούνται αυτές οι τιμές.
      - Αν και οι δύο είναι None: υπολογίζεται αυτόματα από τα whiskers
        (κατώτερο whisker − 1s  έως  ανώτερο whisker + 1s).
    """
    # Κρατάμε μόνο οδηγούς που έχουν δεδομένα στο DataFrame
    drivers_ok = [d.upper() for d in drivers if d.upper() in df["Driver"].unique()]
    if not drivers_ok:
        print("Δεν βρέθηκαν δεδομένα για τους επιλεγμένους οδηγούς.")
        return

    # Λίστα με τους χρόνους γύρου κάθε οδηγού (ένα numpy array ανά οδηγό)
    data = [df.loc[df["Driver"] == d, "LapTime_s"].values for d in drivers_ok]

    # Το πλάτος του γραφήματος κλιμακώνεται με τον αριθμό των οδηγών
    fig, ax = plt.subplots(figsize=(max(8, len(drivers_ok) * 2.5), 6))

    # Τα outliers δεν εμφανίζονται (showfliers=False)
    bp = ax.boxplot(
        data,
        tick_labels=drivers_ok,
        patch_artist=True,       # γεμιστά κουτιά (επιτρέπει χρωματισμό)
        notch=False,
        showfliers=False,        # απόκρυψη outliers
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5, linestyle="none"),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Διαφορετικό χρώμα για κάθε οδηγό
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
               "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Μορφοποίηση άξονα Y: δευτερόλεπτα → M:SS.mmm
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
        # Χειροκίνητο εύρος: μετατροπή "M:SS.mmm" → δευτερόλεπτα
        y_lo = to_seconds(Y_MIN) if Y_MIN else None
        y_hi = to_seconds(Y_MAX) if Y_MAX else None
        ax.set_ylim(bottom=y_lo, top=y_hi)
    else:
        # Αυτόματο εύρος βάσει whiskers (εξαιρούνται outliers).
        # Για κάθε οδηγό υπολογίζουμε:
        #   Q1, Q3 = 25ο και 75ο εκατοστημόριο
        #   IQR = Q3 − Q1
        #   lower fence = Q1 − 1.5 × IQR  |  upper fence = Q3 + 1.5 × IQR
        #   whisker κάτω = ελάχιστη τιμή ≥ lower fence (= κάτω άκρο γραμμής box)
        #   whisker πάνω = μέγιστη τιμή ≤ upper fence (= πάνω άκρο γραμμής box)
        # Τελικό εύρος: [global_min_whisker − 1s,  global_max_whisker + 1s]
        whisker_lo, whisker_hi = [], []
        for d in data:
            arr = np.array(d)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            whisker_lo.append(float(arr[arr >= q1 - 1.5 * iqr].min()))
            whisker_hi.append(float(arr[arr <= q3 + 1.5 * iqr].max()))
        ax.set_ylim(bottom=min(whisker_lo) - 1, top=max(whisker_hi) + 1)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nΔιάγραμμα αποθηκεύτηκε → {output}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PDF: {PDF_PATH}")
    print(f"Οδηγοί: {SELECTED_DRIVERS}\n")

    # Εξαγωγή χρόνων γύρου από το PDF
    laps = extract_laps(PDF_PATH, SELECTED_DRIVERS)

    for driver, lap_list in laps.items():
        print(f"  {driver}: {len(lap_list)} γύροι")

    # Μετατροπή σε DataFrame
    df = build_dataframe(laps)

    print("\nDataFrame (πρώτες 10 γραμμές):")
    print(df.head(10).to_string(index=False))
    print(f"\nΣύνολο γραμμών: {len(df)}")

    # Δημιουργία και αποθήκευση γραφήματος
    plot_boxplot(df, SELECTED_DRIVERS, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
