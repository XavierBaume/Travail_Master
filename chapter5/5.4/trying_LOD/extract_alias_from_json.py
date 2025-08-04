import json
import unicodedata
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple, Dict, Set

Annotation = Tuple[int, int, str, str]  # (start, end, label, uri)

_whitespace_re = re.compile(r"\s+")

def clean_alias(alias: str) -> str:
    alias = unicodedata.normalize("NFKC", alias)
    alias = _whitespace_re.sub(" ", alias)
    return alias.strip()

def load_json(path: Path) -> List[Any]:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de {path} : {e}")
        sys.exit(1)

def extract_aliases(data: List[Any]) -> Dict[str, Set[str]]:
    uri_aliases: Dict[str, Set[str]] = defaultdict(set)
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            continue
        text, annotations = item
        if not isinstance(text, str) or not isinstance(annotations, list):
            continue

        for ann in annotations:
            if isinstance(ann, list) and len(ann) == 4 and isinstance(ann[0], int) and isinstance(ann[1], int) and isinstance(ann[3], str):
                start, end, _label, uri = ann
                if 0 <= start < end <= len(text):
                    alias_text = text[start:end]
                    alias_clean = clean_alias(alias_text)
                    if alias_clean:
                        key = alias_clean.casefold()
                        if key not in {a.casefold() for a in uri_aliases[uri]}:
                            uri_aliases[uri].add(alias_clean)
    return uri_aliases

def save_json(obj: Any, path: Path, description: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ {description} sauvegardé : {path}")

def main() -> None:
    input_path = Path("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/basic_prepared_ner/ner_traindata.json")
    output_path = Path("/Users/xbaume/Documents/MA_Xavier/extract_from_beta_dodis/trying_LOD/recensement_prepared_traindata.json")

    data = load_json(input_path)
    print(f"{len(data)} passages chargés")

    uri_aliases = extract_aliases(data)

    results = [{"uri": uri, "aliases": sorted(aliases)} for uri, aliases in sorted(uri_aliases.items())]
    save_json(results, output_path, "Export")


if __name__ == "__main__":
    main()
