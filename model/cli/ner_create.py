import json
import os
import random
import re
from pathlib import Path

import bs4.element
import typer
from bs4 import BeautifulSoup
import spacy
from spacy.tokens import DocBin


def prepare_links(text: str) -> str:
    return text.replace("http://", "https://")


def load(path: str):
    """
    chargement des fichiers
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def prepare(text: str) -> list:
    """
    Prepare training data pour ner
    Marque les entités avec START:position et END:position et label tei
    """
    # Prétraitement : insérer un espace entre deux liens consécutifs,
    # afin d'éviter la concaténation des textes des balises <a>
    text = re.sub(r'(</a>)(<a\b)', r'\1 \2', text)
    paragraphs = []
    soup = BeautifulSoup(text, 'html.parser')
    title = soup.find('div', {"class": 'tei-title-main'})
    if title is not None:
        paragraphs.append(title)
    sub = soup.find('h1', {"class": 'tei-title-sub'})
    if sub is not None:
        paragraphs.append(sub)
    for p in soup.find_all('p'):
        paragraphs.append(p)
    # format : [texte, [[start, end, label, dodis_id], ...]]
    entities = []
    for p in paragraphs:
        if p is None:
            continue
        entity = Entity()

        # suppression des notes du paragraphe
        [note.extract() for note in p.select('.tei-note4, .tei-note2, .tei-note3, .tei-note')]
        # extraction des entités
        content = p.get_text().strip()
        entity.find(content, p)
        entities.append([content, entity.get_entities()])

    return entities


def trans_class_name(class_name: str) -> str:
    """
    Traduit le nom de classe en label.
    """
    if class_name == "tei-persName":
        return "PER"
    elif class_name == "tei-placeName":
        return "LOC"
    elif class_name == "tei-orgName":
        return "ORG"
    else:
        return "MISC"


class Entity:

    def __init__(self):
        self.entities = []
        self.content = ""
        self.class_names = ["tei-persName", "tei-placeName", "tei-orgName"]

    def find(self, content: str, tag: bs4.element.Tag):
        """
        Recherche toutes les entités TEI dans le HTML.
        """
        self.content = content
        matches = tag.findAll('a')
        if matches is None:
            return

        # pour éviter des chevauchements (ex. PLO et PLO-Büro) ???
        raw = []
        for m in matches:
            if m.get("class") is not None and m.get("class")[0] in self.class_names:
                raw.append((m.get_text().strip(), trans_class_name(m.get("class")[0]), prepare_links(m.get("href"))))

        # suppression des doublons et tri par longueur du plus grand au plus petit
        raw = list(set(raw))
        raw = sorted(raw, reverse=True, key=lambda x: len(x[0]))

        for m, label, dodis_id in raw:
            pattern = r'(%s)' % re.escape(m)
            indexes = re.finditer(pattern, content, flags=re.IGNORECASE) # <--- ajout pour prise en charge des minuscules
            for match in indexes:
                # si un span couvrant déjà cette zone existe, on passe
                found = False
                for ent in self.entities:
                    if ent[0] <= match.start() and ent[1] >= match.end():
                        found = True
                        break
                if not found:
                    self.entities.append((match.start(), match.end(), label, dodis_id))

    def get_entities(self):
        """
        Retourne la liste des entités trouvées
        """
        return [t for t in self.entities if t]


def save_train_data(train_data: list, path: str):
    """
    Sauvegarde les données d'entraînement
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(train_data, f)


def build_traindata(path: Path, target_path: Path):
    """
    Construit les données d'entraînement (train + dev) pour la NER et les sauvegarde au format JSON
    """
    train_data = []
    try:
        entries = os.listdir(path)
        for entry in entries:
            if entry.endswith(".html"):
                data = load(os.path.join(path, entry))
                train_data.extend(prepare(data))
    except FileNotFoundError:
        print(f"Le dossier {path} existe pas.")
    # sauvegarde du format texte déprécié de spacy
    save_train_data(train_data, (target_path / "ner_traindata.json").as_posix())

    # Chargement du modèle spaCy (remarque : on pourrait utiliser notre propre modèle word2vec) ???
    nlp = spacy.load("de_dep_news_trf")
    customize_tokenizer(nlp)
    train_db = DocBin()
    dev_db = DocBin()
    random.shuffle(train_data)
    # répartition train/dev (70% train)
    threshold = len(train_data) * 0.7
    counter = 0

    for text, annotations in train_data:
        if text is None or is_blank(text):
            continue
        doc = nlp(text)
        ents = []
        if annotations is None:
            continue
        for start, end, label, dodis_id in annotations:
            # création du span avec alignment_mode="expand"
            span = doc.char_span(start, end, label=label, kb_id=dodis_id, alignment_mode="expand")
            if span is None:
                print("Impossible de créer un span. Erreur: ", doc.text[start:end], start, end, label)
                continue
            ents.append(span)

        try:
            doc.ents = remove_overlapping(ents)
            if counter < threshold:
                train_db.add(doc)
            else:
                dev_db.add(doc)
        except ValueError as e:
            print(e)
            print("Can't set ents. Error: ", doc.text, ents)
            for ent in ents:
                print(ent.text, ent.label_, ent.start, ent.end)
        counter += 1
    train_db.to_disk((target_path / "ner_traindata.spacy").as_posix())
    dev_db.to_disk((target_path / "ner_devdata.spacy").as_posix())


def is_blank(string: str):
    """
    Vérifie si chaîne vide
    """
    return not (string and string.strip())


def remove_overlapping(ents: list) -> list:
    """
    Supprime les entités qui se chevauchent et corrige les cas d'espaces manquants:
    
    Pour chaque span, si manque d'espace entre maj et min, on "split" via regex.
    Si un span apparaît en tant que concaténation de deux spans contigus issus d'un split
    Si chevauchement, on conserve le span couvrant le plus grand nombre de caractères.
    """
    import re
    missing_space_pattern = r'(?<=[a-z])(?=[A-Z])'
    new_ents = []
    for ent in ents:
        if re.search(missing_space_pattern, ent.text):
            parts = re.split(missing_space_pattern, ent.text)
            if len(parts) > 1:
                print(f"Correction des entités splités '{ent.text}' dans {parts}")
                offset = ent.start_char
                for part in parts:
                    if not part:
                        continue
                    new_end = offset + len(part)
                    new_span = ent.doc.char_span(offset, new_end, label=ent.label_, kb_id=ent.kb_id, alignment_mode="expand")
                    if new_span is None:
                        print(f"Impossible de créer un span pour '{part}' (Position {offset}-{new_end})")
                    else:
                        new_ents.append(new_span)
                    offset = new_end
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    # Suppression des doublons (cf.(start, end, label))
    dedup = {}
    for span in new_ents:
        key = (span.start_char, span.end_char, span.label_)
        dedup[key] = span
    candidates = list(dedup.values())

    # Suppression des spans qui sont collés
    to_remove = set()
    for span in candidates:
        for s1 in candidates:
            for s2 in candidates:
                if s1 is not span and s2 is not span and s1 is not s2:
                    if s1.start_char == span.start_char and s2.end_char == span.end_char and s1.end_char == s2.start_char:
                        if (s1.text + s2.text).replace(" ", "") == span.text.replace(" ", ""):
                            print(f" supression de span concat '{span.text}' pour avoir des spans splités '{s1.text}' and '{s2.text}'.")
                            to_remove.add(span)
    candidates = [span for span in candidates if span not in to_remove]

    # Suppression des chevauchements : si conflit, on garde le span le plus long
    spans = sorted(candidates, key=lambda s: s.start_char)
    keep = [True] * len(spans)
    for i in range(len(spans)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(spans)):
            if not keep[j]:
                continue
            if spans[i].start_char < spans[j].end_char and spans[j].start_char < spans[i].end_char:
                len_i = spans[i].end_char - spans[i].start_char
                len_j = spans[j].end_char - spans[j].start_char
                if len_i >= len_j:
                    print(f"Overlap detecté: '{spans[i].text}' (len={len_i}) recouvre '{spans[j].text}' (len={len_j}), plus long span gardé.")
                    keep[j] = False
                else:
                    print(f"Overlap detecté: '{spans[j].text}' (len={len_j}) recouvre '{spans[i].text}' (len={len_i}), plus long span gardé.")
                    keep[i] = False
                    break
    final = [spans[i] for i in range(len(spans)) if keep[i]]
    return final


def customize_tokenizer(nlp):
    """
    tokenizer personnel pour découper les tokens entre une lettre minuscule et une majuscule pour les spans collés
    """
    import spacy.util
    infixes = list(nlp.Defaults.infixes)
    pattern = r'(?<=[a-z])(?=[A-Z])'
    if pattern not in infixes:
        infixes.append(pattern)
    infix_re = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer


if __name__ == "__main__":
    typer.run(build_traindata)