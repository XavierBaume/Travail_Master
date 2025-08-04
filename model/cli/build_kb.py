import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Sequence

import numpy as np
import spacy
from spacy.tokens import DocBin
from spacy.kb import InMemoryLookupKB
from transformers import AutoModel, AutoTokenizer
import torch

"""
Usage:
python cli/build_kb_from_spacy.py \
    --spacy /Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_train.spacy /Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_nel/ner_nel_dev.spacy \
    --output ./kb_exported
"""

# ------------ CONFIGURATION HF ENCODER -------------
HF_MODEL = "bert-base-german-cased"
_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
_model = AutoModel.from_pretrained(HF_MODEL)
_model.eval()

def embed_batch(texts: List[str], max_length: int = 128) -> np.ndarray:
    with torch.no_grad():
        toks = _tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        outs = _model(**toks).last_hidden_state
        mask = toks["attention_mask"]
        summed = (outs * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return (summed / denom).cpu().numpy()

def clean_alias(alias: str) -> str:
    return alias.strip().replace("\u00A0", " ")

def pad_vec(vec: np.ndarray, dim: int) -> np.ndarray:
    if vec.shape[0] == dim:
        return vec
    if vec.shape[0] > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=vec.dtype)
    out[: vec.shape[0]] = vec
    return out

# ------------- KB BUILDER FROM SPACY ---------------

def build_kb_from_spacy(
    spacy_paths: Sequence[Path],
    kb_dir: Path,
    nlp_model: str = "de_dep_news_trf",
    desc_file: str = "descriptions.json",
    batch_size: int = 32,
    log_level: str = "INFO"
):
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger("spacy_kb_builder")

    nlp = spacy.load(nlp_model, disable=["ner", "entity_linker"])
    kb_dir = Path(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)

    # --- Extraction phase ---
    freq = Counter()
    alias2ent2f = defaultdict(lambda: defaultdict(int))
    desc: Dict[str, str] = {}
    entity_contexts: Dict[str, List[str]] = defaultdict(list)
    n_mentions = 0

    logger.info("Étape 1/4 : extraction contextes et alias depuis .spacy...")
    for path in spacy_paths:
        logger.info(f"  → Fichier : {path}")
        db = DocBin().from_disk(str(path))
        docs = list(db.get_docs(nlp.vocab))
        for doc_idx, doc in enumerate(docs):
            for ent in doc.ents:
                kb_id = ent.kb_id_ if hasattr(ent, "kb_id_") else None
                if not kb_id or not ent.text.strip():
                    continue
                eid = str(kb_id)
                mention = clean_alias(ent.text)
                freq[eid] += 1
                alias2ent2f[mention][eid] += 1
                desc.setdefault(eid, mention)
                # Cherche le contexte : la phrase contenant la mention, ou fallback texte
                sent = next((s for s in doc.sents if s.start_char <= ent.start_char < s.end_char), None)
                context = sent.text if sent else doc.text
                entity_contexts[eid].append(context)
                n_mentions += 1
            if (doc_idx+1) % 10000 == 0:
                logger.info(f"  {doc_idx+1} docs traités...")

    logger.info(f"{n_mentions} mentions extraites")
    logger.info(f"{len(entity_contexts)} entités uniques pour contextualisation")

    # --- Export stats/contrôle ---
    export_stats = {
        "nb_mentions": n_mentions,
        "nb_entities": len(entity_contexts),
        "top_alias": Counter([alias for alias in alias2ent2f]).most_common(10),
        "top_entity": freq.most_common(10)
    }
    (kb_dir / "stats.json").write_text(json.dumps(export_stats, indent=2, ensure_ascii=False), "utf-8")
    logger.info("Stats exportées dans stats.json")
    # Exporte les alias pour audit
    with open(kb_dir / "alias_sample.json", "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k,v in list(alias2ent2f.items())[:20]}, f, indent=2, ensure_ascii=False)

    # --- Export descriptions ---
    desc_path = kb_dir / desc_file
    desc_path.write_text(json.dumps(desc, indent=2, ensure_ascii=False), "utf-8")
    logger.info(f"Descriptions sauvegardées dans {desc_path}")

    # --- Embedding phase ---
    vec_len = _model.config.hidden_size
    logger.info(f"Étape 2/4 : encodage contextuel ({vec_len} dimensions, batch={batch_size})...")
    entity_ids = []
    freqs = []
    vectors = []
    failed_entities = 0

    for eid, contexts in entity_contexts.items():
        if not contexts:
            vec = np.zeros(vec_len, dtype=np.float32)
            logger.warning(f"Aucune occurrence pour {eid} → vecteur nul")
            failed_entities += 1
        else:
            vectors_c = []
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i : i + batch_size]
                try:
                    v = embed_batch(batch)
                    vectors_c.extend(v)
                except Exception as exc:
                    logger.warning(f"Encodage batch {i//batch_size} pour {eid} échoué → {exc}")
            if vectors_c:
                vec = np.mean(vectors_c, axis=0)
            else:
                vec = np.zeros(vec_len, dtype=np.float32)
                logger.warning(f"Aucun embedding valide pour {eid} → vecteur nul")
                failed_entities += 1
        entity_ids.append(eid)
        freqs.append(float(freq[eid]))
        vectors.append(pad_vec(vec, vec_len))

    logger.info(f"{failed_entities} entités sans embeddings valides (vecteur nul)")

    # --- KB Construction phase ---
    logger.info("Étape 3/4 : création de la KB (set_entities)...")
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=vec_len)
    kb.set_entities(entity_ids, freqs, vectors)
    logger.info(f"Ajouté {len(entity_ids)} entités à la KB")

    logger.info("Étape 4/4 : insertion des alias...")
    skipped = 0
    for alias, cand in alias2ent2f.items():
        if not alias:
            skipped += 1
            continue
        ents = list(cand)
        total = sum(cand.values())
        priors = [(cand[e] + 1) / (total + len(ents)) for e in ents]
        try:
            kb.add_alias(alias, ents, priors)
        except Exception as exc:
            skipped += 1
            logger.warning(f"Alias '{alias}' ignoré : {exc}")

    logger.info(f"{skipped} alias ignorés")

    # --- Export KB, meta, test reload ---
    meta = {"vec": vec_len, "model": nlp_model}
    (kb_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    kb.to_disk(kb_dir / "dodis_kb")
    logger.info("KB sauvegardée dans 'dodis_kb'")

    # Test de rechargement
    kb_test = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=vec_len)
    kb_test.from_disk(kb_dir / "dodis_kb")
    print("="*60)
    print("Vérification finale KB spaCy :")
    print(f"  Entités dans KB : {len(list(kb_test.get_entity_strings()))}")
    print(f"  Alias dans KB   : {len(list(kb_test.get_alias_strings()))}")
    print("="*60)
    print("Export(s) de contrôle :")
    print("  - stats.json")
    print("  - alias_sample.json (20 premiers alias/entités)")
    print("  - descriptions.json")
    print("  - meta.json")
    print("  - dodis_kb (dossier KB binaire spaCy)")
    print("="*60)
    print("Script terminé !")

# ----------------- CLI SIMPLE ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génère la KB spaCy depuis des fichiers .spacy (DocBin)")
    parser.add_argument("--spacy", nargs="+", required=True, help="Fichiers .spacy (train/dev...)")
    parser.add_argument("--output", required=True, help="Dossier où écrire la KB et les exports")
    parser.add_argument("--model", default="de_dep_news_trf", help="Modèle spaCy pour le vocabulaire")
    parser.add_argument("--desc-file", default="descriptions.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    spacy_paths = [Path(p) for p in args.spacy]
    build_kb_from_spacy(
        spacy_paths=spacy_paths,
        kb_dir=Path(args.output),
        nlp_model=args.model,
        desc_file=args.desc_file,
        batch_size=args.batch_size,
        log_level=args.log_level
    )
