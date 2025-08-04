import os
import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import nltk

from collections import defaultdict

nltk.download('punkt')

# === Paramètres ===
INPUT_DIR = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/raw_data")
OUTPUT_CSV = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_count_by_lang_exhaustive.csv")
EXPORT_LANG_DIR = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/lang_exports")

# === Config ===
TARGET_LANGS = {'fr': 'français', 'de': 'allemand', 'en': 'anglais', 'it': 'italien'}
ENTITY_CLASSES = ["tei-persName", "tei-placeName", "tei-orgName"]
ENTITY_LABELS = {"tei-persName": "PER", "tei-placeName": "LOC", "tei-orgName": "ORG"}
ID_PATTERNS = {
    "PER": re.compile(r"^https?://dodis\.ch/P\d+$"),
    "LOC": re.compile(r"^https?://dodis\.ch/G\d+$"),
    "ORG": re.compile(r"^https?://dodis\.ch/R\d+$"),
}
FALLBACK_LANG = "inconnue"

# === Stockage des résultats pour le CSV
ALL_LANGS = list(TARGET_LANGS.keys()) + [FALLBACK_LANG]
lang_entity_counts = {l: {lab: 0 for lab in ENTITY_LABELS.values()} for l in ALL_LANGS}
skipped_files = []
skipped_entities = []
skipped_paragraphs = 0
total_paragraphs = 0

# === Stockage pour l'export par langue (NER)
lang_sentences = defaultdict(list)
EXPORT_LANG_DIR.mkdir(exist_ok=True)

for file in INPUT_DIR.glob("*.html"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERREUR] Lecture du fichier {file.name}: {e}")
        skipped_files.append((file.name, str(e)))
        continue

    soup = BeautifulSoup(text, "html.parser")

    # Blocs à traiter = titres et paragraphes
    blocks = []
    title = soup.find('div', {"class": 'tei-title-main'})
    if title: blocks.append(title)
    subtitle = soup.find('h1', {"class": 'tei-title-sub'})
    if subtitle: blocks.append(subtitle)
    ps = soup.find_all('p')
    blocks.extend(ps)

    for p in blocks:
        if p is None:
            continue
        [note.extract() for note in p.select('.tei-note, .tei-note2, .tei-note3, .tei-note4')]
        content = p.get_text().strip()
        total_paragraphs += 1
        if not content:
            continue

        # Détection de la langue
        try:
            lang = detect(content)
            if lang not in TARGET_LANGS:
                lang = FALLBACK_LANG
        except LangDetectException:
            lang = FALLBACK_LANG

        # === Export par langue (NER) ===
        if lang in TARGET_LANGS:  # Ne stocke que les phrases des langues cibles
            # Découpe en phrases individuelles
            nltk_lang = 'french' if lang == 'fr' else ('german' if lang == 'de' else ('english' if lang == 'en' else 'italian'))
            try:
                sentences = nltk.sent_tokenize(content, language=nltk_lang)
            except Exception:
                sentences = [content]  # fallback : pas de découpage
            for sent in sentences:
                sent = sent.strip().replace("\n", " ")
                if sent:
                    lang_sentences[lang].append(sent)

        # === Extraction des entités et comptage ===
        found_entities = set()
        for a in p.find_all('a'):
            try:
                if a.get("class") and a.get("class")[0] in ENTITY_CLASSES:
                    class_name = a.get("class")[0]
                    label = ENTITY_LABELS[class_name]
                    href = a.get("href")
                    alias = a.get_text().strip()
                    if href and ID_PATTERNS[label].match(href.strip()):
                        found_entities.add((alias, label))
                    else:
                        skipped_entities.append(f"[{file.name}] Lien incorrect ou manquant pour entité '{alias}' [{label}]")
            except Exception as e:
                skipped_entities.append(f"[{file.name}] Problème d'entité: {e}")

        if not found_entities:
            skipped_paragraphs += 1
            continue

        for alias, label in found_entities:
            pattern = re.escape(alias)
            matches = re.findall(pattern, content, flags=re.IGNORECASE)
            nb_mentions = len(matches)
            lang_entity_counts[lang][label] += nb_mentions

# === Export CSV ===
try:
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["langue", "type_entite", "nb_mentions"])
        for lang, entity_counts in lang_entity_counts.items():
            lang_display = TARGET_LANGS.get(lang, FALLBACK_LANG)
            for label, count in entity_counts.items():
                writer.writerow([lang_display, label, count])
    print(f"Export terminé : {OUTPUT_CSV}")
except Exception as e:
    print(f"[ERREUR] Problème lors de l'écriture du CSV : {e}")

# === Export fichiers de phrases par langue ===
for lang in TARGET_LANGS:
    lang_name = TARGET_LANGS[lang]
    export_path = EXPORT_LANG_DIR / f"export_{lang}.txt"
    with open(export_path, "w", encoding="utf-8") as f:
        for sentence in lang_sentences[lang]:
            f.write(sentence + "\n")
    print(f"Export des phrases {lang_name} terminé : {export_path}")

# === Récapitulatif des warnings ===
if skipped_files:
    print("\nFichiers ignorés ou problématiques :")
    for fname, reason in skipped_files:
        print(f"  {fname} : {reason}")

if skipped_entities:
    print(f"\nEntités ignorées ou non valides : {len(skipped_entities)} (voir détails ci-dessous)")
    for line in skipped_entities[:10]:
        print(f"  {line}")
    if len(skipped_entities) > 10:
        print(f"  ... ({len(skipped_entities) - 10} autres entités ignorées)")

print(f"\nNombre total de paragraphes (titres inclus) non pris en compte dans l’analyse (=aucune entité détectée) : {skipped_paragraphs}")

if total_paragraphs > 0:
    proportion = (skipped_paragraphs / total_paragraphs) * 100
    print(f"Proportion de paragraphes non pris en compte : {proportion:.2f}% ({skipped_paragraphs}/{total_paragraphs})")
else:
    print("Aucun paragraphe analysé")
