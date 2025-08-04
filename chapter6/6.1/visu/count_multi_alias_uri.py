import json
from collections import Counter, defaultdict

# Remplacer par ton chemin
with open("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_aliases_distribution.json", "r", encoding="utf-8") as f:
    data = json.load(f)

shared = data.get("shared_across_multiple_uris", [])

alias_url_counts = []
for entry in shared:
    alias = entry.get("alias", "")
    uris = entry.get("uris", [])
    num_urls = len(uris)
    alias_url_counts.append((alias, num_urls))

print("\nStatistiques globales :")
aliases_by_count = defaultdict(list)
for alias, nb in alias_url_counts:
    aliases_by_count[nb].append(alias)

for nb_urls in sorted(aliases_by_count):
    alias_list = aliases_by_count[nb_urls]
    print(f"{len(alias_list)} alias avec {nb_urls} URLs :")
    print("    " + ", ".join(alias_list))

all_uris = []
for entry in shared:
    all_uris.extend(entry.get("uris", []))

# Si tu veux supprimer les doublons
unique_uris = set(all_uris)

print(f"\nNombre total d'URLs (avec doublons) : {len(all_uris)}")
print(f"Nombre total d'URLs uniques : {len(unique_uris)}")
