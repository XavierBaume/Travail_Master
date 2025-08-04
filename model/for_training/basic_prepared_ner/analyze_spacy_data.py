import spacy
from spacy.tokens import DocBin
from collections import Counter

def analyze_spacy_file(path):
    print(f"Analyse du fichier : {path}")
    nlp = spacy.blank("xx")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    total_docs = len(docs)
    total_sents = 0
    docs_no_sent = 0
    entity_counter = Counter()
    sample_ents = []

    for doc in docs:
        # Vérifie si l'annotation de segmentation de phrases est présente
        if doc.has_annotation("SENT_START"):
            sents = list(doc.sents)
            total_sents += len(sents)
        else:
            docs_no_sent += 1

        for ent in doc.ents:
            entity_counter[ent.label_] += 1
            if len(sample_ents) < 5:
                sample_ents.append((ent.text, ent.label_))

    print(f"Nombre de documents : {total_docs}")
    print(f"Nombre total de phrases : {total_sents}")

    if docs_no_sent > 0:
        print(f"\n⚠️  Attention : {docs_no_sent} documents (sur {total_docs}) n'ont pas d'annotation de segmentation de phrases (SENT_START).")
        print("    -> Le comptage des phrases peut être incomplet ou incorrect pour ces documents.\n")

    print(f"Nombre total d'entités : {sum(entity_counter.values())}")
    print("Répartition par type d'entité :")
    for label, count in entity_counter.most_common():
        print(f"  {label} : {count}")

    print("\nExemples d'entités extraites :")
    for text, label in sample_ents:
        print(f"  - {text} [{label}]")

# Comparer les différents fichiers .spacy générés au cours des essais de pré-processing
analyze_spacy_file("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_2/ner_devdata.spacy") # ner_devdata.spacy" pour comparaison

