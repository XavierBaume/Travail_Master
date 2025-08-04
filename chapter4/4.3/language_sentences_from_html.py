import os
import glob
import logging
from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt
from collections import Counter
import nltk

nltk.download('punkt')

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dossier source
source_dir = "raw_data"
output_file = "percent_per_langinconnu.txt"

# 1. Recherche des fichiers .txt dans le dossier 'raw_data'
logging.info(f"Recherche des fichiers .txt dans le dossier '{source_dir}'...")
txt_files = glob.glob(os.path.join(source_dir, "*.html"))

if not txt_files:
    logging.error("Aucun fichier .txt trouvé dans le dossier spécifié.")
    exit(1)
else:
    logging.info(f"{len(txt_files)} fichier(s) .txt trouvé(s).")

# ** Nouveau log : Nombre total de documents **
logging.info(f"Nombre total de documents à analyser : {len(txt_files)}")

# Liste pour stocker les langues détectées (par phrase)
detected_langs = []

# ** Ouverture du fichier de sortie pour enregistrer les phrases inconnues **
with open(output_file, "w", encoding="utf-8") as out_file:
    out_file.write("=== Phrases non détectées ===\n\n")

    # 2. Parcours de chaque fichier .txt
    for file_index, file_path in enumerate(txt_files, start=1):
        try:
            # Lire tout le texte du fichier
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Segmentation du texte en phrases
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                logging.warning(f"Aucune phrase détectée dans {file_path}")
                continue
            
            i = 0
            while i < len(sentences):
                sentence = sentences[i]
                try:
                    # Tenter de détecter la langue de la phrase
                    lang = detect(sentence)
                    detected_langs.append(lang)
                except LangDetectException:
                    # Si la détection échoue, concaténer avec la phrase suivante si possible
                    if i + 1 < len(sentences):
                        combined_sentence = sentence + " " + sentences[i + 1]
                        try:
                            lang = detect(combined_sentence)
                            detected_langs.append(lang)
                            i += 1  # Sauter la phrase suivante puisqu'on l'a utilisée
                        except LangDetectException:
                            # Échec après concaténation → sauvegarde dans le fichier d'erreurs
                            logging.warning(f"Langue non détectée même après concaténation : {combined_sentence[:50]}...")
                            out_file.write(f"[Document {file_index} - {os.path.basename(file_path)}] {combined_sentence.strip()}\n\n")
                            detected_langs.append("inconnue")
                    else:
                        # Échec final (dernière phrase) → sauvegarde dans le fichier d'erreurs
                        logging.warning(f"Langue non détectée pour la phrase finale : {sentence[:50]}...")
                        out_file.write(f"[Document {file_index} - {os.path.basename(file_path)}] {sentence.strip()}\n\n")
                        detected_langs.append("inconnue")
                
                i += 1

        except Exception as e:
            logging.error(f"Erreur lors de la lecture de {file_path}: {e}")
            continue

# 3. Comptage et calcul des pourcentages pour les langues d'intérêt
logging.info("Calcul des pourcentages pour les langues cibles...")
target_langs = ['en', 'de', 'it', 'fr']
filtered_langs = [lang for lang in detected_langs if lang in target_langs]
counts = Counter(filtered_langs)

total = sum(counts.values())
if total == 0:
    logging.error("Aucune langue cible détectée dans le corpus.")
    exit(1)

percentages = {lang: (count / total) * 100 for lang, count in counts.items()}

# 4. Affichage des résultats en console
logging.info("Pourcentage par langue :")
for lang, perc in percentages.items():
    logging.info(f"{lang} : {perc:.1f}%")

# 5. Visualisation avec un diagramme circulaire
logging.info("Génération de la visualisation...")
labels = list(percentages.keys())
sizes = list(percentages.values())

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Répartition des langues dans le corpus (par phrase)")
plt.axis('equal')  # Pour assurer que le diagramme est circulaire
plt.show()

plt.savefig("repartition_langues.png", dpi=300, bbox_inches='tight')

logging.info("Visualisation terminée.")

logging.info(f"Fichier 'inconnu.txt' généré avec les phrases non détectées.")
