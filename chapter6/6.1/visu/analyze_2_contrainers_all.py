import spacy
from spacy.tokens import DocBin
from collections import Counter, defaultdict
import json

def extract_entities(path, labels=("ORG", "PER", "LOC")):
    nlp = spacy.blank("xx")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    counters = {label: Counter() for label in labels}
    examples = {label: defaultdict(list) for label in labels}

    for doc in docs:
        for ent in doc.ents:
            if ent.label_ in labels:
                counters[ent.label_][ent.text] += 1
                if len(examples[ent.label_][ent.text]) < 2:
                    examples[ent.label_][ent.text].append(doc.text[:80])  # contexte

    return counters, examples

def build_entity_report(label, train_counter, train_examples, dev_counter, dev_examples):
    total_counter = train_counter + dev_counter
    train_set = set(train_counter)
    dev_set = set(dev_counter)

    unique_entities = [
        {
            "text": ent,
            "total_count": 1,
            "in_train": int(ent in train_counter),
            "in_dev": int(ent in dev_counter),
            "context": train_examples.get(ent, dev_examples.get(ent, [""])).pop()
        }
        for ent, count in total_counter.items() if count == 1
    ]

    only_in_train = sorted(list(train_set - dev_set))
    only_in_dev = sorted(list(dev_set - train_set))
    in_both = sorted(list(train_set & dev_set))

    return {
        "summary": {
            f"train_total_{label.lower()}s": sum(train_counter.values()),
            f"train_unique_{label.lower()}s": len(train_counter),
            f"dev_total_{label.lower()}s": sum(dev_counter.values()),
            f"dev_unique_{label.lower()}s": len(dev_counter),
            f"total_unique_{label.lower()}s_corpus": len(train_set | dev_set),
            f"{label.lower()}s_appearing_once_total": len(unique_entities),
            f"{label.lower()}s_only_in_train": len(only_in_train),
            f"{label.lower()}s_only_in_dev": len(only_in_dev),
            f"{label.lower()}s_in_both": len(in_both),
        },
        f"{label.lower()}s_appearing_once_details": unique_entities,
        f"{label.lower()}s_only_in_train": only_in_train,
        f"{label.lower()}s_only_in_dev": only_in_dev,
        f"{label.lower()}s_in_both": in_both,
    }

# === Chemins ===
train_path = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_traindata.spacy"
dev_path = "/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_devdata.spacy"

# === Extraction ===
labels = ("ORG", "PER", "LOC")
train_counters, train_examples = extract_entities(train_path, labels)
dev_counters, dev_examples = extract_entities(dev_path, labels)

# === Construction de l'export global ===
export = {}

for label in labels:
    report = build_entity_report(
        label,
        train_counters[label],
        train_examples[label],
        dev_counters[label],
        dev_examples[label]
    )
    export[label] = report

# === Export JSON ===
with open("/Users/xbaume/Documents/MA_Xavier/extract_from_beta_dodis/all_entities_structured_report.json", "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
