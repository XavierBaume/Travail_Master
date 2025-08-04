import spacy
from spacy.kb import InMemoryLookupKB

nlp = spacy.blank("de")
kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=768)
kb.from_disk("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/kb_exported/dodis_kb")

print("Entités dans la KB :", len(list(kb.get_entity_strings())))
print("Alias dans la KB   :", len(list(kb.get_alias_strings())))


entity_ids = list(kb.get_entity_strings())
if entity_ids:
    ent_id = entity_ids[0]
    embedding = kb.get_vector(ent_id)
    print("Longueur de l'embedding:", len(embedding))
else:
    print("Pas d'entité dans la KB !")