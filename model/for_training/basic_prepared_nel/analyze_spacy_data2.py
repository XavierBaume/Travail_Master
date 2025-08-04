import spacy
from spacy.tokens import DocBin
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

def analyze_nel_datasets(path_train, path_dev):
    print(f"Chargement : {path_train}")
    train_db = DocBin().from_disk(path_train)
    print(f"Chargement : {path_dev}")
    dev_db = DocBin().from_disk(path_dev)

    def extract_info(doc):
        ents = doc.ents
        links = doc.user_data.get("links", {})
        data = []
        for ent in ents:
            # liens = doc.user_data['links'][(ent.start_char, ent.end_char)] si présent
            key = (ent.start_char, ent.end_char)
            kb_ids = []
            if key in links:
                kb_ids = list(links[key].keys())
            elif hasattr(ent, "kb_id_") and ent.kb_id_:
                kb_ids = [ent.kb_id_]
            else:
                kb_ids = []
            data.append({
                "text": ent.text,
                "label": ent.label_,
                "kb_id": "|".join(kb_ids) if kb_ids else "",
                "span": (ent.start_char, ent.end_char)
            })
        return data

    stats = {
        "train": [],
        "dev": [],
    }
    for s, db in [("train", train_db), ("dev", dev_db)]:
        docs = list(db.get_docs(spacy.blank("de").vocab))
        for i, doc in enumerate(docs):
            info = extract_info(doc)
            for ent in info:
                ent['doc_id'] = i
                stats[s].append(ent)
        print(f"{s}: {len(docs)} docs | {len(stats[s])} entités extraites")

    def show_stats(name, ents):
        print(f"\n==== Stats pour {name} ====")
        df = pd.DataFrame(ents)
        print("Total docs :", len(set(df['doc_id'])))
        print("Total entités annotées :", len(df))
        print("Total spans uniques :", len(set((e['doc_id'], e['span']) for e in ents)))
        print("Classes :", dict(Counter(df['label'])))
        print("Entités sans kb_id :", (df['kb_id'] == "").sum())
        print("Entités par kb_id :", df['kb_id'].nunique(), " (sur ", len(df), ")")
        print("Entités singletons (1 seule occurrence):", (df['kb_id'].value_counts() == 1).sum())
        print("Entités présentes plusieurs fois :", (df['kb_id'].value_counts() > 1).sum())
        print("10 kb_id les plus fréquents :", df['kb_id'].value_counts().head(10))
        print("10 mentions les plus fréquentes :", df['text'].value_counts().head(10))
        # Détection spans chevauchants
        spans = [(row['doc_id'], row['span']) for row in ents]
        if len(spans) != len(set(spans)):
            print("⚠️ Chevauchements de span détectés dans ce split !")
        # Signalement entités vides
        if (df['text'].str.strip() == "").any():
            print("⚠️ Entités avec texte vide détectées !")
        print("Exemples entités sans kb_id :")
        print(df[df['kb_id'] == ""].head(3))

    show_stats("train", stats["train"])
    show_stats("dev", stats["dev"])

    # Croisement entités entre train/dev
    train_kb = set([e['kb_id'] for e in stats['train'] if e['kb_id']])
    dev_kb = set([e['kb_id'] for e in stats['dev'] if e['kb_id']])
    print("\nIntersection kb_id train/dev :", len(train_kb & dev_kb))
    print("kb_id uniquement dans train :", len(train_kb - dev_kb))
    print("kb_id uniquement dans dev :", len(dev_kb - train_kb))

    # Suggestions
    if (train_kb & dev_kb):
        print("✅ Certaines entités sont bien présentes dans les deux splits (attendu pour NEL).")
    if "" in train_kb or "" in dev_kb:
        print("⚠️ Présence d'entités sans kb_id : elles seront ignorées par la NEL.")
    rare_in_dev = [kb for kb in dev_kb if stats['dev'].count({'kb_id': kb}) == 1]
    if len(rare_in_dev) > 0:
        print("⚠️ Plusieurs kb_id apparaissent une seule fois dans le dev (singletons).")
    # Idéalement, vérifier aussi la diversité des contextes (non traité ici)

if __name__ == "__main__":
    # Adapte les chemins selon ta structure
    analyze_nel_datasets(
        path_train="/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_train.spacy",
        path_dev="/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_dev.spacy"
    )