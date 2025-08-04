import spacy
from spacy.tokens import DocBin
from collections import Counter

def analyze_spacy_file(path, focus=None):
    print(f"Analyse du fichier : {path}")
    nlp = spacy.blank("xx")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    total_docs = len(docs)
    total_sents = 0
    docs_no_sent = 0
    entity_counter = Counter()
    sample_ents = []
    focused_ents = []

    for doc_id, doc in enumerate(docs):
        if doc.has_annotation("SENT_START"):
            sents = list(doc.sents)
            total_sents += len(sents)
        else:
            docs_no_sent += 1

        for ent in doc.ents:
            entity_counter[ent.label_] += 1
            if len(sample_ents) < 5:
                sample_ents.append((ent.text, ent.label_))
            # Focalisation sur une forme orthographique
            if focus and focus.lower() in ent.text.lower():
                # Récupération de la phrase contenant l'entité
                sent_text = ent.sent.text if hasattr(ent, 'sent') else "[Phrase indisponible]"
                focused_ents.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "doc_id": doc_id,
                    "sent": sent_text
                })

    print(f"Nombre de documents : {total_docs}")
    print(f"Nombre total de phrases : {total_sents}")

    if docs_no_sent > 0:
        print(f"\n⚠️  Attention : {docs_no_sent} documents (sur {total_docs}) n'ont pas d'annotation de segmentation de phrases (SENT_START).")
        print("    -> Le comptage des phrases peut être incomplet ou incorrect pour ces documents.\n")

    print(f"Nombre total d'entités : {sum(entity_counter.values())}")
    print("Répartition par type d'entité :")
    for label, count in entity_counter.most_common():
        print(f"  {label} : {count}")

    if focus:
        print(f"\nEntités contenant « {focus} » :")
        if focused_ents:
            for ent in focused_ents:
                print(f"  - {ent['text']} [{ent['label']}] (doc {ent['doc_id']}, positions {ent['start_char']}-{ent['end_char']})")
                print(f"    → Phrase : {ent['sent']}")
            print(f"\nNombre total d'occurrences pour « {focus} » : {len(focused_ents)}")
            
            # Comptage par catégorie d'entité
            focused_counter = Counter([ent['label'] for ent in focused_ents])
            print("Répartition par type d'entité pour le focus :")
            for label, count in focused_counter.most_common():
                print(f"  {label} : {count}")
        else:
            print(f"Aucune entité ne contient « {focus} ».")

# Exemple d'appel avec focus
analyze_spacy_file(
    "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_traindata.spacy",
    focus="Arpad Göncz"
)
