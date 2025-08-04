import spacy
import csv
import json
import re
from collections import defaultdict, Counter

def normalize_entity(text):
    # Normalisation simple : minuscule, suppression pluriel/suffixes de type -es/-e, stripping
    text = text.strip().lower()
    text = re.sub(r"^(der|die|das|le|la|les|the)\s+", "", text)
    # Pluriel allemand simple
    text = re.sub(r"(e|es|er|n|en|s)$", "", text)
    # Spécifique pour les cas 'Bundesrat/Bundesrates/Bundesrats...'
    text = re.sub(r"(bundesrat(s|es|beschluss|beschlüsse|ssitzung)?)", "bundesrat", text)
    return text

PATH_NER = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/best-model-50553975"
PATH_NEL = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/el-results/model-best"

nlp = spacy.load(PATH_NER)
nlp.add_pipe("entity_linker", source=spacy.load(PATH_NEL), last=True)

INPUT_JSON = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_traindata.json"
CSV_EXPORT = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/test-2/results_org_entities_goldonly_enriched.csv"
CSV_TOP_FP = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/test-2/top_fp_normalized.csv"
CSV_MULTI_FP_PHRASE = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/test-2/phrases_multi_fp.csv"
CSV_HOMONYMY = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/test-2/homonymes_kb_id.csv"

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
entity_contexts = defaultdict(list)         # Pour chaque entité, ses contextes
entity_kbids = defaultdict(set)             # Pour chaque entité, ses kb_id
phrase_fp_counts = defaultdict(list)        # Pour chaque phrase, liste des faux positifs
entity_stats = Counter()                    # Pour stats globales
entity_lengths = []                         # Pour stats sur longueur

for idx, entry in enumerate(data, 1):
    phrase = entry[0]
    annots = entry[1] if len(entry) > 1 else []
    has_org_gold = any(ann[2] == "ORG" for ann in annots)
    if not has_org_gold:
        continue

    org_expected = set()
    for ann in annots:
        label = ann[2]
        if label == "ORG":
            span = phrase[ann[0]:ann[1]]
            kid = ann[3] if len(ann) > 3 else ""
            org_expected.add((span, label, kid))

    doc = nlp(phrase)
    org_predicted = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_predicted.add((ent.text, ent.label_, ent.kb_id_))

    matched = org_expected & org_predicted
    missed_by_model = org_expected - org_predicted
    extra_predicted = org_predicted - org_expected

    # --- Enrichissement et stats ---
    norm_predicted = []
    for span, label, kid in extra_predicted:
        norm = normalize_entity(span)
        entity_stats[norm] += 1
        entity_lengths.append(len(span))
        entity_contexts[norm].append({"phrase": phrase, "entity_text": span, "kb_id": kid})
        entity_kbids[norm].add(kid)
        phrase_fp_counts[phrase].append((span, kid))

    # --- Export enrichi ---
    if matched:
        for span, label, kid in matched:
            rows.append({
                "phrase": phrase,
                "entity_text": span,
                "norm_entity": normalize_entity(span),
                "label": label,
                "kb_id": kid,
                "source": "ATTENDU+PREDIT",
                "statut": "MATCH"
            })
    if missed_by_model:
        for span, label, kid in missed_by_model:
            rows.append({
                "phrase": phrase,
                "entity_text": span,
                "norm_entity": normalize_entity(span),
                "label": label,
                "kb_id": kid,
                "source": "ATTENDU",
                "statut": "ABSENT_DANS_PREDICTION"
            })
    if extra_predicted:
        for span, label, kid in extra_predicted:
            rows.append({
                "phrase": phrase,
                "entity_text": span,
                "norm_entity": normalize_entity(span),
                "label": label,
                "kb_id": kid,
                "source": "PREDIT",
                "statut": "ABSENT_DANS_ANNOTATION"
            })

# -- Export principal enrichi --
with open(CSV_EXPORT, "w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["phrase", "entity_text", "norm_entity", "label", "kb_id", "source", "statut"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"\nExport principal enrichi : {CSV_EXPORT} ({len(rows)} lignes)")

# -- Export : top faux positifs normalisés (familles d’erreurs) --
top_fp = entity_stats.most_common(30)
with open(CSV_TOP_FP, "w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["norm_entity", "nb_fp", "kb_ids", "exemples"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ent, nb in top_fp:
        # Limite à 3 contextes typiques pour annotation rapide
        exemples = [ctx["entity_text"] for ctx in entity_contexts[ent][:3]]
        writer.writerow({
            "norm_entity": ent,
            "nb_fp": nb,
            "kb_ids": "|".join(entity_kbids[ent]),
            "exemples": " | ".join(exemples)
        })
print(f"Export top faux positifs (familles d’erreurs) : {CSV_TOP_FP}")

# -- Export : phrases avec beaucoup de faux positifs (cas emblématiques) --
multi_fp_phrases = [(phrase, fps) for phrase, fps in phrase_fp_counts.items() if len(fps) >= 3]
multi_fp_phrases = sorted(multi_fp_phrases, key=lambda x: -len(x[1]))[:20]
with open(CSV_MULTI_FP_PHRASE, "w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["phrase", "nb_fp", "faux_positifs"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for phrase, fps in multi_fp_phrases:
        writer.writerow({
            "phrase": phrase,
            "nb_fp": len(fps),
            "faux_positifs": " | ".join([f"{e[0]} ({e[1]})" for e in fps])
        })
print(f"Export phrases à forte densité de faux positifs : {CSV_MULTI_FP_PHRASE}")

# -- Export : entités homonymes (plusieurs KB_ID pour la même forme) --
homonymes = [(ent, kbids) for ent, kbids in entity_kbids.items() if len(kbids) > 1]
with open(CSV_HOMONYMY, "w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["norm_entity", "kb_ids"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ent, kbids in homonymes:
        writer.writerow({
            "norm_entity": ent,
            "kb_ids": "|".join(kbids)
        })
print(f"Export homonymies : {CSV_HOMONYMY}")

# -- Stats globales --
print("\n=== STATISTIQUES QUALITATIVES ENRICHIES ===")
print(f"Nombre total de faux positifs (ORG, normalisé) : {sum(entity_stats.values())}")
print(f"Nombre d’entités ORG différentes (normalisées) : {len(entity_stats)}")
print("Top 10 des familles d’erreurs (entités normalisées les plus fréquentes) :")
for ent, nb in top_fp[:10]:
    print(f"{ent} : {nb}")

import numpy as np
if entity_lengths:
    arr = np.array(entity_lengths)
    print("\nLongueur moyenne des entités faussement détectées : %.2f caractères (min : %d, max : %d)" % (arr.mean(), arr.min(), arr.max()))

print("\nExport enrichi terminé. Analyse qualitative prête pour annotation et exploration historique.")
