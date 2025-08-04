import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

# === CONFIGURATION DES RÉPERTOIRES ===
JSON_DIR = Path('/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/metadata/json-extracted-structured')
HTML_DIR = Path('/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/raw_data')
OUTPUT_DIR = Path("outputs_cross_entities_HTML_META")
OUTPUT_DIR.mkdir(exist_ok=True)

# === REGEX POUR HTML (URL Dodis attendues) ===
ID_PATTERNS = {
    "PER": re.compile(r"^https?://dodis\.ch/P\d+$"),
    "LOC": re.compile(r"^https?://dodis\.ch/G\d+$"),
    "ORG": re.compile(r"^https?://dodis\.ch/R\d+$"),
}

# === MAPPINGS DE CLASSES ===
def trans_class_name(class_name: str) -> str:
    return {"tei-persName": "PER", "tei-placeName": "LOC", "tei-orgName": "ORG"}.get(class_name, "MISC")

FIELD_TO_CLASS = {
    "authors": "PER",
    "adressee": "PER",
    "signatories": "PER",
    "mentioned_persons": "PER",
    "org_authors": "ORG",
    "org_adressee": "ORG",
    "org_mentioned": "ORG",
    "places_mentioned": "LOC",
    "places_destination": "LOC"
}

# === EXTRACTION JSON (avec conversion en URLs normalisées) ===
json_entities = defaultdict(set)
for file in JSON_DIR.glob("*.json"):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for field, ent_class in FIELD_TO_CLASS.items():
                for item in data.get(field, []):
                    if isinstance(item, dict) and 'id' in item:
                        base_id = str(item['id']).strip()
                        if not base_id:
                            continue
                        if ent_class == "PER":
                            entity_id = f"https://dodis.ch/P{base_id}"
                        elif ent_class == "ORG":
                            entity_id = f"https://dodis.ch/R{base_id}"
                        elif ent_class == "LOC":
                            entity_id = f"https://dodis.ch/G{base_id}"
                        else:
                            continue
                        json_entities[ent_class].add(entity_id)
    except Exception as e:
        print(f"Erreur dans {file.name}: {e}")

# === EXTRACTION HTML (identifiants TEI déjà normalisés) ===
html_entities = defaultdict(set)
for file in HTML_DIR.glob("*.html"):
    with open(file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    elements = []
    title = soup.find('div', class_='tei-title-main')
    if title: elements.append(title)
    subtitle = soup.find('h1', class_='tei-title-sub')
    if subtitle: elements.append(subtitle)
    elements.extend(soup.find_all('p'))

    for el in elements:
        if el is None: continue
        [note.extract() for note in el.select('.tei-note4, .tei-note2, .tei-note3, .tei-note')]
        for a in el.find_all('a'):
            href = a.get("href")
            classes = a.get("class", [])
            for cls in classes:
                label = trans_class_name(cls)
                if label in ID_PATTERNS and href and ID_PATTERNS[label].match(href):
                    html_entities[label].add(href)

# === ANALYSE COMPARATIVE ===
comparison_data = []

common_all, only_json_all, only_html_all = [], [], []

with open(OUTPUT_DIR / "entities_common.csv", "w", newline='', encoding="utf-8") as f1, \
     open(OUTPUT_DIR / "entities_only_json.csv", "w", newline='', encoding="utf-8") as f2, \
     open(OUTPUT_DIR / "entities_only_html.csv", "w", newline='', encoding="utf-8") as f3:

    writer_common = csv.writer(f1)
    writer_only_json = csv.writer(f2)
    writer_only_html = csv.writer(f3)

    writer_common.writerow(["Type", "Entity"])
    writer_only_json.writerow(["Type", "Entity"])
    writer_only_html.writerow(["Type", "Entity"])

    print("\n=== COMPARAISON ENTRE JSON ET HTML ===")
    for ent_class in ["PER", "LOC", "ORG"]:
        json_set = json_entities[ent_class]
        html_set = html_entities[ent_class]
        common = json_set & html_set
        only_json = json_set - html_set
        only_html = html_set - json_set

        common_all.append(len(common))
        only_json_all.append(len(only_json))
        only_html_all.append(len(only_html))

        print(f"\n--- {ent_class} ---")
        print(f"JSON      : {len(json_set)} entités uniques")
        print(f"HTML/TEI  : {len(html_set)} entités uniques")
        print(f"COMMUNES  : {len(common)}")
        print(f"Seulement JSON : {len(only_json)}")
        print(f"Seulement HTML : {len(only_html)}")

        if html_set:
            print(f"Taux de recouvrement HTML dans JSON : {len(common)/len(html_set):.2%}")
        if json_set:
            print(f"Taux de recouvrement JSON dans HTML : {len(common)/len(json_set):.2%}")

        for e in common:
            writer_common.writerow([ent_class, e])
        for e in only_json:
            writer_only_json.writerow([ent_class, e])
        for e in only_html:
            writer_only_html.writerow([ent_class, e])

        comparison_data.append({
            "class": ent_class,
            "json": len(json_set),
            "html": len(html_set),
            "common": len(common),
            "only_json": len(only_json),
            "only_html": len(only_html)
        })

# === RÉSUMÉ GLOBAL ===
total_common = sum(common_all)
total_json = sum(len(s) for s in json_entities.values())
total_html = sum(len(s) for s in html_entities.values())

print("\n=== RÉSUMÉ GLOBAL ===")
print(f"Total JSON      : {total_json}")
print(f"Total HTML/TEI  : {total_html}")
print(f"Total COMMUNES  : {total_common}")
if total_html:
    print(f"Taux de chevauchement global (HTML → JSON) : {total_common / total_html:.2%}")
if total_json:
    print(f"Taux de chevauchement global (JSON → HTML) : {total_common / total_json:.2%}")

# === VISUALISATION ===
classes = [d["class"] for d in comparison_data]
bar_width = 0.25
x = range(len(classes))

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width for i in x], [d["only_json"] for d in comparison_data], width=bar_width, label="Uniquement JSON")
plt.bar(x, [d["common"] for d in comparison_data], width=bar_width, label="Communes")
plt.bar([i + bar_width for i in x], [d["only_html"] for d in comparison_data], width=bar_width, label="Uniquement HTML")

plt.xlabel("Classe d'entités")
plt.ylabel("Nombre d'entités uniques")
plt.title("Comparaison des entités entre JSON et HTML/TEI")
plt.xticks(ticks=x, labels=classes)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recap_barplot.png")
plt.close()
