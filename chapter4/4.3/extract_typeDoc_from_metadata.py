import os
import json
from collections import Counter

# Dossier contenant les fichiers .json
folder_path = '/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/metadata/json-extracted-structured'

# Initialisation du compteur pour les types de documents
type_counter = Counter()

# Parcours de tous les fichiers dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                type_text = data.get('type_text')
                if type_text:
                    type_counter[type_text] += 1
        except (json.JSONDecodeError, IOError) as e:
            print(f"Erreur avec le fichier {filename}: {e}")

# Affichage des résultats
print("Répartition des 'type_text' dans les documents JSON :")
for type_text, count in type_counter.most_common():
    print(f"{type_text}: {count}")
