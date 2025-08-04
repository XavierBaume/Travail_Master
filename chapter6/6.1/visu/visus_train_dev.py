import matplotlib.pyplot as plt
import numpy as np

# Données
labels = ['ORG', 'PER', 'LOC']
train_counts = [7879, 23736, 49633]
dev_counts = [3297, 10148, 20657]
unique_train = [2332, 6383, 4993]
unique_dev = [1151, 3704, 2854]
common_train_dev = [545, 2159, 1674]
singletons = [1993, 4458, 3685]
total_unique = [
    unique_train[0] + unique_dev[0] - common_train_dev[0],
    unique_train[1] + unique_dev[1] - common_train_dev[1],
    unique_train[2] + unique_dev[2] - common_train_dev[2]
]

# 1. Occurrences par type d'entité (train/dev)
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, train_counts, width, label='Train')
ax.bar(x + width/2, dev_counts, width, label='Dev')
ax.set_ylabel("Nombre d'occurrences")
ax.set_title("Occurrences par type d'entité")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/visu/occurrences_par_type.png")
plt.close()

# 2. Entités uniques par type (train/dev)
fig, ax = plt.subplots()
ax.bar(x - width/2, unique_train, width, label='Unique train')
ax.bar(x + width/2, unique_dev, width, label='Unique dev')
ax.set_ylabel("Nombre d'entités uniques")
ax.set_title("Entités uniques par type d'entité")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/visu/entites_uniques_par_type.png")
plt.close()

# 3. Taux de recouvrement entre train et dev
overlap_percent = [c/d*100 for c, d in zip(common_train_dev, unique_dev)]
fig, ax = plt.subplots()
ax.bar(labels, overlap_percent)
ax.set_ylabel("Taux de recouvrement (%)")
ax.set_title("Taux de recouvrement des entités uniques (train ∩ dev / dev)")
plt.tight_layout()
plt.savefig("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/visu/taux_recouvrement.png")
plt.close()

# 4. Proportion d'entités uniques (singleton)
singleton_ratio = [s/t*100 for s, t in zip(singletons, total_unique)]
fig, ax = plt.subplots()
ax.bar(labels, singleton_ratio)
ax.set_ylabel("Proportion d'entités uniques (%)")
ax.set_title("Proportion d'entités n'apparaissant qu'une seule fois")
plt.tight_layout()
plt.savefig("/Users/xbaume/Documents/MA_Xavier/NER/fancy-ml-ner_xav/results_ner_nel/visu/proportion_singletons.png")
plt.close()

print("Visualisations générées et enregistrées avec succès !")
