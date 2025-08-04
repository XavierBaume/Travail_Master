import os
import re
import json
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import bs4.element

# === Répertoires ===
INPUT_DIR = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/raw_data")
OUTPUT_COUNT_FILE = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_count.txt")
OUTPUT_WARNINGS_FILE = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_warnings.txt")
OUTPUT_ALIAS_JSON = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_aliases.json")
OUTPUT_ALIAS_DISTRIBUTION_JSON = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_aliases_distribution.json")
OUTPUT_SINGLETON_CSV = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_singletons.csv")

# === Compteurs et collecteurs ===
entity_counter = {"PER": 0, "LOC": 0, "ORG": 0}
entity_unique_ids = {"PER": set(), "LOC": set(), "ORG": set()}
entity_aliases = {}
ignored_entities = []
alias_to_uris = {}
uri_mention_counts = {}

# === Regex pour validation des liens ===
ID_PATTERNS = {
    "PER": re.compile(r"^https?://dodis\.ch/P\d+$"),
    "LOC": re.compile(r"^https?://dodis\.ch/G\d+$"),
    "ORG": re.compile(r"^https?://dodis\.ch/R\d+$"),
}

def trans_class_name(class_name: str) -> str:
    return {
        "tei-persName": "PER",
        "tei-placeName": "LOC",
        "tei-orgName": "ORG"
    }.get(class_name, "MISC")

class Entity:
    def __init__(self):
        self.entities = []
        self.content = ""
        self.class_names = ["tei-persName", "tei-placeName", "tei-orgName"]

    def find(self, content: str, tag: bs4.element.Tag):
        self.content = content
        matches = tag.find_all('a')
        raw = []
        for m in matches:
            if m.get("class") and m.get("class")[0] in self.class_names:
                label = trans_class_name(m.get("class")[0])
                text = m.get_text().strip()
                href = m.get("href")
                if not href:
                    ignored_entities.append(f"[{file.name}] Entité sans href : {text}")
                    continue
                href = re.sub(r"^http://", "https://", href.strip())
                raw.append((text, label, href))
        #raw = list(set(raw))  # suppression des doublons
        raw = sorted(set(raw))  # suppression des doublons

        for m, label, href in raw:
            pattern = r'(%s)' % re.escape(m)
            indexes = re.finditer(pattern, content, flags=re.IGNORECASE)
            for match in indexes:
                found = False
                for ent in self.entities:
                    if ent[0] <= match.start() and ent[1] >= match.end():
                        found = True
                        break
                if not found:
                    self.entities.append((match.start(), match.end(), label, href))
                    # Enregistrement des alias
                    if href in entity_aliases:
                        entity_aliases[href]["aliases"].add(m)
                    else:
                        entity_aliases[href] = {
                            "label": label,
                            "aliases": {m}
                        }
                    # Mapping alias → URIs
                    if m in alias_to_uris:
                        alias_to_uris[m].add(href)
                    else:
                        alias_to_uris[m] = {href}
                    # Comptage des mentions par URI
                    if href in uri_mention_counts:
                        uri_mention_counts[href] += 1
                    else:
                        uri_mention_counts[href] = 1

    def get_entities(self):
        return [t for t in self.entities if t]

# === Traitement de chaque fichier HTML ===
#for file in INPUT_DIR.glob("*.html"):
for file in sorted(INPUT_DIR.glob("*.html")):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    soup = BeautifulSoup(text, "html.parser")
    paragraphs = []

    title = soup.find('div', {"class": 'tei-title-main'})
    if title:
        paragraphs.append(title)
    subtitle = soup.find('h1', {"class": 'tei-title-sub'})
    if subtitle:
        paragraphs.append(subtitle)
    paragraphs.extend(soup.find_all('p'))

    for p in paragraphs:
        if p is None:
            continue
        [note.extract() for note in p.select('.tei-note, .tei-note2, .tei-note3, .tei-note4')]
        content = p.get_text().strip()
        entity = Entity()
        entity.find(content, p)
        for start, end, label, href in entity.get_entities():
            entity_counter[label] += 1
            if ID_PATTERNS.get(label) and ID_PATTERNS[label].match(href):
                entity_unique_ids[label].add(href)
            else:
                ignored_entities.append(f"[{file.name}] Lien incorrect ou inattendu pour {label}: {href}")

# === Résultats texte ===
total_mentions = sum(entity_counter.values())
total_uniques = sum(len(s) for s in entity_unique_ids.values())

with open(OUTPUT_COUNT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Comptage global des entités extraites ===\n")
    for label, count in entity_counter.items():
        f.write(f"{label}: {count} mentions\n")
    f.write(f"Total des mentions: {total_mentions}\n\n")

    f.write("=== Comptage des entités différentes (unitaires) ===\n")
    for label, unique_set in entity_unique_ids.items():
        f.write(f"{label}: {len(unique_set)} identifiants uniques\n")
    f.write(f"Total des entités différentes: {total_uniques}\n")

with open(OUTPUT_WARNINGS_FILE, "w", encoding="utf-8") as f:
    if ignored_entities:
        f.write("=== Entités ignorées ou problématiques ===\n")
        for line in ignored_entities:
            f.write(line + "\n")
    else:
        f.write("Aucune entité ignorée.\n")

# === Export JSON des URI et alias ===
exported_aliases = {
    uri: {
        "label": data["label"],
        "aliases": sorted(data["aliases"])
    }
    for uri, data in entity_aliases.items()
}

with open(OUTPUT_ALIAS_JSON, "w", encoding="utf-8") as f:
    json.dump(exported_aliases, f, indent=2, ensure_ascii=False)

# === Export JSON des alias selon leur distribution sur les URI ===
alias_distribution = {
    "unique_to_single_uri": [],
    "shared_across_multiple_uris": [],
    "totals": {
        "unique_to_single_uri": 0,
        "shared_across_multiple_uris": 0,
    }
}

for alias, uris in alias_to_uris.items():
    if len(uris) == 1:
        alias_distribution["unique_to_single_uri"].append({
            "alias": alias,
            "uri": list(uris)[0]
        })
        alias_distribution["totals"]["unique_to_single_uri"] += 1
    else:
        alias_distribution["shared_across_multiple_uris"].append({
            "alias": alias,
            "uris": sorted(uris)
        })
        alias_distribution["totals"]["shared_across_multiple_uris"] += 1

with open(OUTPUT_ALIAS_DISTRIBUTION_JSON, "w", encoding="utf-8") as f:
    json.dump(alias_distribution, f, indent=2, ensure_ascii=False)

# === Export CSV des entités apparaissant une seule fois (singleton entities) ===
with open(OUTPUT_SINGLETON_CSV, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "uri", "aliases"])
    for uri, count in uri_mention_counts.items():
        if count == 1:
            label = entity_aliases.get(uri, {}).get("label", "")
            aliases = ", ".join(sorted(entity_aliases.get(uri, {}).get("aliases", [])))
            writer.writerow([label, uri, aliases])

print(f"Comptage terminé. Résultats dans :\n- {OUTPUT_COUNT_FILE}\n- {OUTPUT_WARNINGS_FILE}\n- {OUTPUT_ALIAS_JSON}\n- {OUTPUT_ALIAS_DISTRIBUTION_JSON}\n- {OUTPUT_SINGLETON_CSV}")
