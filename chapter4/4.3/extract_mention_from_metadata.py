import os
import json
from collections import defaultdict

# Dossier contenant les fichiers JSON
input_dir = '/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/metadata/json-extracted-structured'

# Dictionnaire pour stocker les IDs uniques par type d'entité
entity_ids_by_class = defaultdict(set)

# Mapping des champs vers leur type d'entité
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

# Parcours des fichiers JSON
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        path = os.path.join(input_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for field, ent_class in FIELD_TO_CLASS.items():
                    if field in data and isinstance(data[field], list):
                        for item in data[field]:
                            if isinstance(item, dict) and 'id' in item:
                                ent_id = item['id']
                                if isinstance(ent_id, (str, int)) and ent_id:
                                    entity_ids_by_class[ent_class].add(ent_id)

        except Exception as e:
            print(f"Erreur dans {filename}: {e}")

# Affichage du nombre total d'entités uniques par classe
for ent_class, ids in entity_ids_by_class.items():
    print(f"{ent_class} : {len(ids)} entités uniques")
