import spacy
from spacy.tokens import DocBin

# Charger le fichier .spacy
nlp = spacy.blank("xx")
doc_bin = DocBin().from_disk("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_traindata.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Extraire un seul Doc, par exemple le 42e
doc = docs[42]

# Affichage
print("\nTexte du document :")
print(doc.text)

print("\nEntités annotées :")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})", f"kb_id={ent.kb_id}")