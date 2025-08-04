import re
from pathlib import Path
from bs4 import BeautifulSoup

# Répertoire contenant les fichiers HTML
INPUT_DIR = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/raw_data")
OUTPUT_FILE = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/find_washington/washington_references.txt")

# Classe TEI → Label
def trans_class_name(class_name: str) -> str:
    if class_name == "tei-persName":
        return "PER"
    elif class_name == "tei-placeName":
        return "LOC"
    elif class_name == "tei-orgName":
        return "ORG"
    else:
        return "MISC"

# Résultats
results = []
counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0, "UNKNOWN": 0}

# Analyse des fichiers
for file in INPUT_DIR.glob("*.html"):
    with open(file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for a in soup.find_all('a'):
        if a.text.strip().lower() == "washington":
            classes = a.get("class", [])
            href = a.get("href", "N/A")
            label = "UNKNOWN"

            for cls in classes:
                if cls in ["tei-persName", "tei-placeName", "tei-orgName"]:
                    label = trans_class_name(cls)
                    break

            counts[label] = counts.get(label, 0) + 1

            result = {
                "fichier": file.name,
                "texte": a.text.strip(),
                "classe": label,
                "href": href
            }
            results.append(result)

# Écriture du fichier de sortie
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    if results:
        f.write("=== Références à 'Washington' ===\n")
        for r in results:
            f.write(f"[{r['fichier']}] ({r['classe']}) → {r['texte']} → {r['href']}\n")

        f.write("\n=== Statistiques des occurrences ===\n")
        for label in ["PER", "LOC", "ORG", "MISC", "UNKNOWN"]:
            f.write(f"{label}: {counts[label]} occurrence(s)\n")
    else:
        f.write("Aucune occurrence de 'Washington' trouvée.\n")

print(f"✅ Analyse terminée. Résultats dans : {OUTPUT_FILE}")
