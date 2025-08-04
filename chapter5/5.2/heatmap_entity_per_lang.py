import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/entity_count_by_lang_exhaustive.csv')

# Table des pourcentages
pivot = df.pivot(index='type_entite', columns='langue', values='nb_mentions').fillna(0)
pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100

plt.figure(figsize=(8, 4))

cmap = sns.color_palette("Blues", as_cmap=True)

sns.heatmap(
    pivot_percent,
    annot=True,
    fmt=".1f",
    cmap=cmap,
    cbar_kws={"label": "% de mentions"}
)
plt.title("")
plt.ylabel("Type d'entité")
plt.xlabel("Langue du paragraphe")
plt.tight_layout()
plt.savefig('/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/data_pre_analysis/heatmap_entity_lang.png')
plt.close()
