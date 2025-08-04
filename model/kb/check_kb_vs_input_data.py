import spacy
from spacy.kb import InMemoryLookupKB
from spacy.tokens import DocBin

# === Chemins ===
KB_PATH = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/kb_exported/dodis_kb"
TRAIN_PATH = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_train.spacy"
DEV_PATH = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_dev.spacy"
ENTITY_VECTOR_LENGTH = 768  # À adapter si besoin

# === Charger la KB ===
nlp = spacy.blank("de")
kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=ENTITY_VECTOR_LENGTH)
kb.from_disk(KB_PATH)
kb_ent_ids = set(kb.get_entity_strings())

print(f"Nombre d’entités dans la KB           : {len(kb_ent_ids)}")

# === Fonction pour collecter les entités annotées dans un corpus ===
def collect_ent_ids(spacy_file):
    doc_bin = DocBin().from_disk(spacy_file)
    ent_ids = set()
    for doc in doc_bin.get_docs(nlp.vocab):
        for ent in doc.ents:
            if ent.kb_id_:
                ent_ids.add(ent.kb_id_)
    return ent_ids

# === Charger les entités annotées dans train/dev ===
ent_ids_train = collect_ent_ids(TRAIN_PATH)
ent_ids_dev = collect_ent_ids(DEV_PATH)
corpus_ent_ids = ent_ids_train | ent_ids_dev

print(f"Nombre d’entités annotées dans corpus : {len(corpus_ent_ids)}")

# === Comparaison ===
missing_in_kb = corpus_ent_ids - kb_ent_ids
unused_in_corpus = kb_ent_ids - corpus_ent_ids

if missing_in_kb:
    print(f"\n {len(missing_in_kb)} entités annotées dans le corpus MANQUENT dans la KB ! (erreur critique)")
    print("Quelques exemples :", list(missing_in_kb)[:10])
else:
    print("\n Toutes les entités annotées dans le corpus sont présentes dans la KB.")

print(f"\n {len(unused_in_corpus)} entités dans la KB ne sont JAMAIS annotées dans le corpus.")
if unused_in_corpus:
    print("Quelques exemples :", list(unused_in_corpus)[:10])
