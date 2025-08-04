import json
import csv

# Chemin vers le fichier JSON
input_file = "/Users/xbaume/Documents/MA_Xavier/extract_from_beta_dodis/trying_LOD/recensement_prepared_traindata.json"
output_file = "/Users/xbaume/Documents/MA_Xavier/extract_from_beta_dodis/trying_LOD/uris_uniques.csv"

# Charger les données JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraire les URIs uniques
unique_uris = sorted({entry["uri"] for entry in data if "uri" in entry})

# Exporter en CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["uri"])  # En-tête
    for uri in unique_uris:
        writer.writerow([uri])

# Application de la logique de "Find - Replace" pour obtenir la forme "https://beta.dodis.ch/""