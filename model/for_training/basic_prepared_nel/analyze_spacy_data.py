import spacy
from spacy.tokens import DocBin
from collections import Counter
from pathlib import Path

def analyze_spacy_file(spacy_path, nb_examples=5):
    print(f"\n📁 Analyse de : {spacy_path}")
    doc_bin = DocBin().from_disk(spacy_path)
    docs = list(doc_bin.get_docs(spacy.blank("de").vocab))
    print(f"Nombre total de docs : {len(docs)}")

    total_ents = 0
    label_counter = Counter()
    kb_counter = Counter()
    examples = []

    for doc in docs:
        ents = doc.ents
        total_ents += len(ents)
        for ent in ents:
            label_counter[ent.label_] += 1
            kb_counter[ent.kb_id_] += 1
            # On retient quelques exemples avec texte, label, kb_id, start, end
            if len(examples) < nb_examples:
                examples.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "kb_id": ent.kb_id_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        # Vérification de la présence du champ links
        if "links" in doc.user_data and len(doc.user_data["links"]) > 0 and len(examples) < nb_examples:
            for (start, end), candidates in doc.user_data["links"].items():
                print(f"  Lien NEL : {doc.text[start:end]} --> {candidates}")

    print(f"Nombre total d'entités : {total_ents}")
    print("Répartition par label :")
    for label, count in label_counter.items():
        print(f"   {label}: {count}")
    print(f"Nombre de kb_id différents : {len(kb_counter)}")
    print("\nExemples d'entités annotées :")
    for ex in examples:
        print(f" - '{ex['text']}' [{ex['label']}] (kb_id: {ex['kb_id']}) {ex['start']}-{ex['end']}")

    # Affiche quelques kb_id les plus fréquents (debug)
    print("\nkb_id les plus fréquents :")
    for kb, cnt in kb_counter.most_common(5):
        print(f"   {kb}: {cnt}")

if __name__ == "__main__":
    train_path = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_train.spacy")
    dev_path = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_dev.spacy")
    for spacy_file in [train_path, dev_path]:
        if spacy_file.exists():
            analyze_spacy_file(spacy_file)
        else:
            print(f"Fichier {spacy_file} non trouvé !")
