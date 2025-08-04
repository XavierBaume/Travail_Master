import csv
from collections import defaultdict

# Colonnes à vérifier
target_columns = [
    "wikidata", "wikidata_id", "dhs", "lonsea", "gnd",
    "elites_suisses", "helveticat", "viaf", "huygens", "bhs",
    "eth_bibliothek", "parl_ch", "geonames", "tgn", "sds"
]

# Fichier CSV à analyser
csv_file = "/Users/xbaume/Documents/MA_Xavier/extract_from_beta_dodis/trying_LOD/dodis_links.csv"

# Compteurs globaux
empty_count = 0
column_counts = defaultdict(int)
entity_type_counts = defaultdict(int)

# Compteurs par type d'entité
entity_type_detail = {
    "PER": {"total": 0, "non_empty": 0, "empty": 0},
    "ORG": {"total": 0, "non_empty": 0, "empty": 0},
    "LOC": {"total": 0, "non_empty": 0, "empty": 0}
}

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row.get("url", "").strip()
        entity_type = None

        if url.startswith("https://beta.dodis.ch/"):
            suffix = url.replace("https://beta.dodis.ch/", "").strip()
            if suffix:
                prefix = suffix[0].upper()
                if prefix == "P":
                    entity_type = "PER"
                elif prefix == "R":
                    entity_type = "ORG"
                elif prefix == "G":
                    entity_type = "LOC"

        if entity_type:
            entity_type_counts[entity_type] += 1
            entity_type_detail[entity_type]["total"] += 1

            has_non_empty = any(row.get(col, "").strip() for col in target_columns)
            if has_non_empty:
                entity_type_detail[entity_type]["non_empty"] += 1
            else:
                entity_type_detail[entity_type]["empty"] += 1

            if not has_non_empty:
                empty_count += 1

            for col in target_columns:
                if row.get(col, "").strip():
                    column_counts[col] += 1

# Résultats globaux
print(f"\nNombre de lignes avec une URL mais tous les champs cibles vides : {empty_count}\n")

print("Nombre de lignes avec un champ NON vide par colonne :")
for col in target_columns:
    print(f"  {col}: {column_counts[col]}")

print("\nDétail par type d'entité (PER, ORG, LOC) :")
for etype, stats in entity_type_detail.items():
    print(f"  {etype}: total = {stats['total']}, avec champ rempli = {stats['non_empty']}, vides = {stats['empty']}")
