# -*- coding: utf-8 -*-
"""
САОБРАЋАЈНЕ НЕЗГОДЕ - Предикција тежине исхода
Јована Минчић, ИН4/2023

Садржај:
    1. Библиотеке
    2. Генератор вредности
    3. Учитавање и основно прегледање података
    4. Истражна анализа - недостајуће вредности и основне статистике
    5. Истражна анализа - расподела циљног обележја 
    6. Истражна анализа - расподела категорије незгоде
    7. Истражна анализа - метеоролошки услови
    8. Истражна анализа - временска анализа
    9. Истражна анализа - просторна анализа
    10. Истражна анализа - корелациона анализа
    11. Припрема података за кластеризацију
    12. Кластеризација - Elbow метода
    13. Кластеризација - Silhouette анализа
    14. Кластеризација - K-means алгоритам
    15. Анализа добијених кластера
    16. Припрема података за неуронску мрежу
    17. Поређење оптимизатора (Adam, SGD, Adagrad)
    18. Вишеслојна неуронска мрежа
    19. Евалуација модела
"""


# %% 1. Библиотеке

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,
                             confusion_matrix,
                             classification_report,
                             f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score)

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, Adagrad

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})

# Помоћна функција:
    
DIGRAFI = {"Dž": "Џ", "DŽ": "Џ", "dž": "џ", "Lj": "Љ", "LJ": "Љ", "lj": "љ",
           "Nj": "Њ", "NJ": "Њ", "nj": "њ", "Dj": "Ђ", "DJ": "Ђ", "dj": "ђ"}
SLOVA = {"A": "А", "B": "Б", "C": "Ц", "Č": "Ч", "Ć": "Ћ", "D": "Д", "Đ": "Ђ",
         "E": "Е", "F": "Ф", "G": "Г", "H": "Х", "I": "И", "J": "Ј", "K": "К",
         "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С",
         "Š": "Ш", "T": "Т", "U": "У", "V": "В", "Z": "З", "Ž": "Ж",
         "a": "а", "b": "б", "c": "ц", "č": "ч", "ć": "ћ", "d": "д", "đ": "ђ",
         "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "ј", "k": "к",
         "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с",
         "š": "ш", "t": "т", "u": "у", "v": "в", "z": "з", "ž": "ж"}


def cir(tekst):
    tekst = str(tekst)
    for lat, c in DIGRAFI.items():
        tekst = tekst.replace(lat, c)
    return "".join(SLOVA.get(ch, ch) for ch in tekst)


# %% 2. Генератор вредности

generator = np.random.default_rng(59)


# %% 3. Учитавање и основно прегледање података

skup = pd.read_csv("Nezgode 2025 - meteo.csv")

skup["datum_vreme"] = pd.to_datetime(skup["datum_vreme"],
                                      format="%d.%m.%Y,%H:%M",
                                      errors="coerce")

skup["mesec"] = skup["datum_vreme"].dt.month
skup["dan_u_nedelji"] = skup["datum_vreme"].dt.dayofweek
skup["sat"] = skup["datum_vreme"].dt.hour


godisnja_doba_preslikavanje = {12: 0, 1: 0, 2: 0,
                                3: 1, 4: 1, 5: 1,
                                6: 2, 7: 2, 8: 2,
                                9: 3, 10: 3, 11: 3}
skup["godisnje_doba"] = skup["mesec"].map(godisnja_doba_preslikavanje)
# 0=зима, 1=пролеће, 2=лето, 3=јесен

print("Број редова и колона скупа података: ", skup.shape)
print()
print("Првих 5 редова:")
print(skup.head())
print()
print("Типови података:")
print(skup.dtypes)


# %% 4. Истражна анализа - недостајуће вредности и основне статистике

print("Недостајуће вредности по колонама:")
print(skup.isnull().sum())
print()

print("Основна статистика нумеричких обележја:")
print(skup[["temperature_2m", "precipitation",
            "wind_speed_10m", "cloud_cover",
            "weather_code"]].describe().round(2))


# %% 5. Истражна анализа - расподела циљног обележја 

raspodela_tezina = skup["tezina"].value_counts()
print("Расподела тежине исхода незгоде:")
print(raspodela_tezina)
print()
print("Процентуална расподела:")
print((raspodela_tezina / len(skup) * 100).round(2))


raspodela_tezina_cir = raspodela_tezina.rename(index=cir)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

raspodela_tezina_cir.plot(kind="bar", ax=ax[0],
                          color=["steelblue", "orange", "crimson"],
                          edgecolor="black")
ax[0].set_title("Расподела тежине исхода незгоде")
ax[0].set_xlabel("Тежина")
ax[0].set_ylabel("Број незгода")
ax[0].tick_params(axis="x", rotation=20)
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height()):,}",
                   (p.get_x() + p.get_width()/2, p.get_height()),
                   ha="center", va="bottom", fontsize=10)

raspodela_tezina_cir.plot(kind="pie", ax=ax[1], autopct="%1.1f%%",
                          colors=["steelblue", "orange", "crimson"],
                          startangle=140)
ax[1].set_title("Процентуална расподела тежине")
ax[1].set_ylabel("")

plt.tight_layout()
plt.show()


# %% 6. Истражна анализа - расподела категорије незгоде

raspodela_kategorija = skup["kategorija"].value_counts()
print("Расподела категорија незгода:")
print(raspodela_kategorija)

raspodela_kategorija_cir = raspodela_kategorija.rename(index=cir)
fig, ax = plt.subplots(figsize=(11, 5))
raspodela_kategorija_cir.plot(kind="barh", ax=ax,
                              color="steelblue", edgecolor="black")
ax.set_title("Расподела категорија саобраћајних незгода")
ax.set_xlabel("Број незгода")
ax.invert_yaxis()
for p in ax.patches:
    ax.annotate(f"{int(p.get_width()):,}",
                (p.get_width(), p.get_y() + p.get_height()/2),
                ha="left", va="center", fontsize=10)
plt.tight_layout()
plt.show()


# %% 6b. Веза између категорије и тежине незгоде


tabela_kat_tez = pd.crosstab(skup["kategorija"], skup["tezina"])
print("Апсолутне вредности:")
print(tabela_kat_tez)
print()


tabela_kat_tez_pct = pd.crosstab(skup["kategorija"], skup["tezina"],
                                  normalize="index") * 100
print("Процентуално (по категорији, %):")
print(tabela_kat_tez_pct.round(2))

tabela_kat_tez_cir = tabela_kat_tez.rename(index=cir, columns=cir)
tabela_kat_tez_pct_cir = tabela_kat_tez_pct.rename(index=cir, columns=cir)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))


tabela_kat_tez_cir.plot(kind="barh", stacked=True, ax=axes[0],
                        color=["steelblue", "crimson", "orange"],
                        edgecolor="black")
axes[0].set_title("Тежина исхода по категорији незгоде (апсолутно)")
axes[0].set_xlabel("Број незгода")
axes[0].set_ylabel("Категорија")
axes[0].legend(title="Тежина", loc="lower right")


tabela_kat_tez_pct_cir.plot(kind="barh", stacked=True, ax=axes[1],
                            color=["steelblue", "crimson", "orange"],
                            edgecolor="black")
axes[1].set_title("Тежина исхода по категорији незгоде (%)")
axes[1].set_xlabel("Проценат (%)")
axes[1].set_ylabel("")
axes[1].legend(title="Тежина", loc="lower right")
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()


# %% 7. Истражна анализа - метеоролошки услови

meteo_kolone = ["temperature_2m", "precipitation",
                "wind_speed_10m", "cloud_cover"]
meteo_nazivi = ["Температура [°C]", "Падавине [mm]",
                "Брзина ветра [km/h]", "Облачност [%]"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, kolona, naziv in zip(axes.flatten(), meteo_kolone, meteo_nazivi):
    ax.hist(skup[kolona], bins=40, color="steelblue",
            edgecolor="black", alpha=0.8)
    ax.set_title(f"Расподела: {naziv}")
    ax.set_xlabel(naziv)
    ax.set_ylabel("Фреквенција")
    ax.axvline(skup[kolona].mean(), color="red", linestyle="--",
               linewidth=1.5,
               label=f"Средња вредност: {skup[kolona].mean():.2f}")
    ax.legend()
plt.suptitle("Расподеле метеоролошких обележја", fontsize=14, y=1.00)
plt.tight_layout()
plt.show()


# %% 8. Истражна анализа - временска анализа

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


mesec_nazivi = ["Јан", "Феб", "Мар", "Апр", "Мај", "Јун",
                "Јул", "Авг", "Сеп", "Окт", "Нов", "Дец"]
po_mesecu = skup.groupby("mesec")["id"].count()
axes[0].bar(po_mesecu.index, po_mesecu.values,
            color="steelblue", edgecolor="black")
axes[0].set_title("Број незгода по месецу")
axes[0].set_xlabel("Месец")
axes[0].set_ylabel("Број незгода")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(mesec_nazivi)


po_satu = skup.groupby("sat")["id"].count()
axes[1].plot(po_satu.index, po_satu.values, marker="o",
             color="steelblue", linewidth=2)
axes[1].fill_between(po_satu.index, po_satu.values,
                     alpha=0.2, color="steelblue")
axes[1].set_title("Број незгода по сату у току дана")
axes[1].set_xlabel("Сат")
axes[1].set_ylabel("Број незгода")
axes[1].set_xticks(range(0, 24))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


dani_nazivi = ["Пон", "Уто", "Сре", "Чет", "Пет", "Суб", "Нед"]
fig, ax = plt.subplots(figsize=(11, 5))
tabela_dani = skup.groupby(["dan_u_nedelji", "tezina"])["id"].count(
    ).unstack(fill_value=0)
tabela_dani.index = dani_nazivi
tabela_dani.rename(columns=cir).plot(kind="bar", ax=ax, edgecolor="black",
                                     color=["steelblue", "orange", "crimson"])
ax.set_title("Тежина незгоде по дану у недељи")
ax.set_xlabel("Дан у недељи")
ax.set_ylabel("Број незгода")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Тежина", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %% 8b. Проценат погинулих по дану у недељи
proc_pog = skup.groupby("dan_u_nedelji")["tezina"].apply(
    lambda x: (x == "Sa poginulim").sum() / len(x) * 100)
proc_pog.index = dani_nazivi
print("Проценат погинулих по дану (%):")
print(proc_pog.round(3))

fig, ax = plt.subplots(figsize=(10, 5))
proc_pog.plot(kind="bar", ax=ax, color="crimson", edgecolor="black")
ax.set_title("Проценат смртних исхода по дану у недељи")
ax.set_xlabel("Дан у недељи")
ax.set_ylabel("Проценат погинулих (%)")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()


# %% 9. Истражна анализа - просторна анализа

# Топ 10 општина
top10_opstine = skup["opstina"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
top10_opstine.rename(index=cir).plot(kind="barh", ax=ax,
                                     color="steelblue", edgecolor="black")
ax.set_title("Топ 10 општина по броју незгода")
ax.set_xlabel("Број незгода")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Просторна расподела
boje_tezina = {"Sa mat.stetom": "steelblue",
               "Sa povredjenim": "orange",
               "Sa poginulim": "crimson"}

fig, ax = plt.subplots(figsize=(11, 8))
for tezina, boja in boje_tezina.items():
    podskup = skup[skup["tezina"] == tezina]
    ax.scatter(podskup["lon"], podskup["lat"],
               c=boja, label=cir(tezina), s=2, alpha=0.3)
ax.set_title("Просторна расподела незгода по тежини исхода")
ax.set_xlabel("Географска дужина")
ax.set_ylabel("Географска ширина")
ax.legend(markerscale=5, title="Тежина")
plt.tight_layout()
plt.show()


# %% 10. Истражна анализа - корелациона анализа


tezina_redosled = {"Sa mat.stetom": 0,
                   "Sa povredjenim": 1,
                   "Sa poginulim": 2}
tezina_nazivi = ["Са мат.штетом", "Са повређеним", "Са погинулим"]
skup["tezina_kod"] = skup["tezina"].map(tezina_redosled)

print("Кодовање тежине:")
for naziv, kod in tezina_redosled.items():
    print(f"  {kod} : {naziv}")
print()


le_kategorija = LabelEncoder()
skup["kategorija_kod"] = le_kategorija.fit_transform(skup["kategorija"])


num_kolone = ["temperature_2m", "precipitation", "wind_speed_10m",
              "cloud_cover", "weather_code", "sat", "mesec",
              "dan_u_nedelji", "tezina_kod", "kategorija_kod"]
num_nazivi = ["Температура", "Падавине", "Брзина ветра",
              "Облачност", "Метео код", "Сат", "Месец",
              "Дан у недељи", "Тежина", "Категорија"]

korelaciona_matrica = skup[num_kolone].corr()
print("Корелациона матрица:")
print(korelaciona_matrica.round(3))


fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(korelaciona_matrica, cmap="coolwarm",
               vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(num_kolone)))
ax.set_yticks(range(len(num_kolone)))
ax.set_xticklabels(num_nazivi, rotation=45, ha="right")
ax.set_yticklabels(num_nazivi)


for i in range(len(num_kolone)):
    for j in range(len(num_kolone)):
        ax.text(j, i, f"{korelaciona_matrica.iloc[i, j]:.2f}",
                ha="center", va="center",
                color="black" if abs(
                    korelaciona_matrica.iloc[i, j]) < 0.5 else "white",
                fontsize=8)

plt.colorbar(im, ax=ax)
ax.set_title("Корелациона матрица нумеричких обележја")
plt.tight_layout()
plt.show()


# %% 11. Припрема података за кластеризацију

 
obelezja_klaster = ["lat", "lon", "sat", "mesec",
                    "temperature_2m",
                    "wind_speed_10m", "cloud_cover", "weather_code"]

skup_klaster = skup[obelezja_klaster].copy()
skup_klaster.dropna(inplace=True)


print(f"Број редова за кластеризацију: {len(skup_klaster)}")
print(f"Коришћена обележја: {obelezja_klaster}")
print()


scaler = StandardScaler()
skup_klaster_norm = pd.DataFrame(
    scaler.fit_transform(skup_klaster),
    columns=skup_klaster.columns,
    index=skup_klaster.index)

print("Статистика нормализованих података:")
print(skup_klaster_norm.describe().round(3))


# %% 12. Кластеризација - Elbow метода

k_raspon = range(2, 11)
inercije = []

print("Рачунање инерција за Elbow методу:")
for k in k_raspon:
    km = KMeans(n_clusters=k, random_state=59,
                n_init=10, max_iter=300)
    km.fit(skup_klaster_norm)
    inercije.append(km.inertia_)
    print(f"  K={k}: инерција = {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_raspon), inercije, "bo-",
        linewidth=2, markersize=10)
ax.set_title("Elbow метода за избор оптималног броја кластера")
ax.set_xlabel("Број кластера (K)")
ax.set_ylabel("Инерција (WCSS)")
ax.set_xticks(list(k_raspon))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% 13. Кластеризација - Silhouette анализа

silhouette_skori = []

print("Рачунање silhouette скора:")
for k in k_raspon:
    km = KMeans(n_clusters=k, random_state=59,
                n_init=10, max_iter=300)
    oznake_k = km.fit_predict(skup_klaster_norm)
    skor = silhouette_score(skup_klaster_norm, oznake_k)
    silhouette_skori.append(skor)
    print(f"  K={k}: silhouette = {skor:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_raspon), silhouette_skori, "gs-",
        linewidth=2, markersize=10)
ax.set_title("Silhouette анализа за избор оптималног броја кластера")
ax.set_xlabel("Број кластера (K)")
ax.set_ylabel("Silhouette скор")
ax.set_xticks(list(k_raspon))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


k_optimalno = list(k_raspon)[np.argmax(silhouette_skori)]
print(f"\nОптимални број кластера: K = {k_optimalno}")


# %% 14. Кластеризација - K-means

km_finalno = KMeans(n_clusters=k_optimalno, random_state=59,
                    n_init=10, max_iter=300)
oznake_klastera = km_finalno.fit_predict(skup_klaster_norm)

skup["klaster"] = oznake_klastera

print("Број примера по кластеру:")
print(skup["klaster"].value_counts().sort_index())


# %% 15. Анализа добијених кластера


fig, ax = plt.subplots(figsize=(11, 5))
tabela_klaster = skup.groupby(["klaster", "tezina"])["id"].count(
    ).unstack(fill_value=0)
tabela_klaster.rename(columns=cir).plot(kind="bar", ax=ax, edgecolor="black",
                                        color=["steelblue", "orange", "crimson"])
ax.set_title("Расподела тежине незгоде по кластерима")
ax.set_xlabel("Кластер")
ax.set_ylabel("Број незгода")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Тежина", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


print("Просечне вредности обележја по кластеру:")
print(skup.groupby("klaster")[obelezja_klaster].mean().round(2))


fig, ax = plt.subplots(figsize=(11, 8))
boje_klaster = plt.cm.tab10(np.linspace(0, 1, k_optimalno))
for k in range(k_optimalno):
    podskup_k = skup[skup["klaster"] == k]
    ax.scatter(podskup_k["lon"], podskup_k["lat"],
               c=[boje_klaster[k]], label=f"Кластер {k}",
               s=2, alpha=0.4)
ax.set_title("Просторна расподела кластера")
ax.set_xlabel("Географска дужина")
ax.set_ylabel("Географска ширина")
ax.legend(markerscale=5, title="Кластер")
plt.tight_layout()
plt.show()


# %% 16. Припрема података за неуронску мрежу


kontinualna_obelezja = ["temperature_2m", "cloud_cover",
                        "sat", "mesec", "lat", "lon",
                        "kategorija_kod", "klaster"]

weather_dummies = pd.get_dummies(skup["weather_code"],
                                  prefix="weather").astype(float)
ulaz_sve = pd.concat([skup[kontinualna_obelezja], weather_dummies], axis=1)

broj_ulaza = ulaz_sve.shape[1]
ciljno_obelezje = "tezina_kod"
broj_klasa = skup[ciljno_obelezje].nunique()

print(f"Број улазних обележја: {broj_ulaza}")
print(f"  континуалних: {len(kontinualna_obelezja)}, "
      f"one-hot колона за weather_code: {weather_dummies.shape[1]}")
print(f"Циљно обележје: {ciljno_obelezje}")
print(f"Број класа: {broj_klasa}")


mapa_kod_naziv = dict(enumerate(tezina_nazivi))
print(f"\nМапирање класа: {mapa_kod_naziv}")


idx_obuc, idx_val = train_test_split(
    skup.index, test_size=0.2, random_state=59,
    stratify=skup[ciljno_obelezje])

skup_obucavanje_ulaz = ulaz_sve.loc[idx_obuc].copy()
skup_validacija_ulaz = ulaz_sve.loc[idx_val].copy()


min_trening = skup_obucavanje_ulaz[kontinualna_obelezja].min()
max_trening = skup_obucavanje_ulaz[kontinualna_obelezja].max()
skup_obucavanje_ulaz[kontinualna_obelezja] = (
    skup_obucavanje_ulaz[kontinualna_obelezja] - min_trening) / (max_trening - min_trening)
skup_validacija_ulaz[kontinualna_obelezja] = (
    skup_validacija_ulaz[kontinualna_obelezja] - min_trening) / (max_trening - min_trening)


cilj_obuc = skup.loc[idx_obuc, ciljno_obelezje]
cilj_val = skup.loc[idx_val, ciljno_obelezje]
skup_obucavanje_cilj = to_categorical(cilj_obuc, broj_klasa)
skup_validacija_cilj = to_categorical(cilj_val, broj_klasa)

print(f"\nВеличина скупа за обучавање: {len(idx_obuc)}")
print(f"Величина скупа за валидацију: {len(idx_val)}")
print("Расподела класа у валидацији:")
print(cilj_val.value_counts().sort_index())


# %% 17. Поређење оптимизатора (Adam, SGD, Adagrad)

optimizatori = {
    "Adam":    Adam(learning_rate=0.001),
    "SGD":     SGD(learning_rate=0.01, momentum=0.9),
    "Adagrad": Adagrad(learning_rate=0.01)
}

istorija_opt = {}
broj_epoha = 50

for naziv_opt, opt in optimizatori.items():
    print(f"\n{'='*50}")
    print(f"Обучавање са оптимизатором: {naziv_opt}")
    print(f"{'='*50}")

    mreza_test = Sequential()
    mreza_test.add(Input(shape=(broj_ulaza,)))
    mreza_test.add(Dense(32, activation="relu"))
    mreza_test.add(Dense(16, activation="tanh"))
    mreza_test.add(Dense(broj_klasa, activation="softmax"))

    mreza_test.compile(loss="categorical_crossentropy",
                       optimizer=opt,
                       metrics=["categorical_accuracy"])

    istorija = mreza_test.fit(skup_obucavanje_ulaz, skup_obucavanje_cilj,
                              epochs=broj_epoha,
                              validation_data=(skup_validacija_ulaz,
                                               skup_validacija_cilj),
                              verbose=0)
    istorija_opt[naziv_opt] = istorija.history

    val_acc = istorija.history["val_categorical_accuracy"][-1]
    val_loss = istorija.history["val_loss"][-1]
    print(f"Завршна валидациона тачност: {val_acc:.4f}")
    print(f"Завршни валидациони губитак: {val_loss:.4f}")

# Графичко поређење оптимизатора
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
boje_opt = {"Adam": "steelblue", "SGD": "orange", "Adagrad": "green"}

for naziv_opt in optimizatori:
    axes[0].plot(istorija_opt[naziv_opt]["val_loss"],
                 label=naziv_opt, color=boje_opt[naziv_opt],
                 linewidth=2)
    axes[1].plot(istorija_opt[naziv_opt]["val_categorical_accuracy"],
                 label=naziv_opt, color=boje_opt[naziv_opt],
                 linewidth=2)

axes[0].set_title("Валидациони губитак - поређење оптимизатора")
axes[0].set_xlabel("Епоха")
axes[0].set_ylabel("Губитак")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Валидациона тачност - поређење оптимизатора")
axes[1].set_xlabel("Епоха")
axes[1].set_ylabel("Тачност")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% 18. Коначна неуронска мрежа


print("Обучавање коначне неуронске мреже (Adam оптимизатор)")
print("="*60)

mreza = Sequential()
mreza.add(Input(shape=(broj_ulaza,)))
mreza.add(Dense(64, activation="relu"))
mreza.add(Dropout(0.3))
mreza.add(Dense(32, activation="tanh"))
mreza.add(Dense(16, activation="relu"))
mreza.add(Dense(broj_klasa, activation="softmax"))

mreza.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["categorical_accuracy"])

mreza.summary()

istorija_finalno = mreza.fit(skup_obucavanje_ulaz, skup_obucavanje_cilj,
                             epochs=80,
                             batch_size=64,
                             validation_data=(skup_validacija_ulaz,
                                              skup_validacija_cilj),
                             verbose=1)

# Криве обучавања
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(istorija_finalno.history["loss"],
             label="Тренинг губитак", color="steelblue", linewidth=2)
axes[0].plot(istorija_finalno.history["val_loss"],
             label="Валидациони губитак", color="crimson",
             linestyle="--", linewidth=2)
axes[0].set_title("Крива губитка током обучавања")
axes[0].set_xlabel("Епоха")
axes[0].set_ylabel("Губитак")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(istorija_finalno.history["categorical_accuracy"],
             label="Тренинг тачност", color="steelblue", linewidth=2)
axes[1].plot(istorija_finalno.history["val_categorical_accuracy"],
             label="Валидациона тачност", color="crimson",
             linestyle="--", linewidth=2)
axes[1].set_title("Крива тачности током обучавања")
axes[1].set_xlabel("Епоха")
axes[1].set_ylabel("Тачност")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% 19. Евалуација модела

print("="*60)
print("ЕВАЛУАЦИЈА МОДЕЛА")
print("="*60)

mreza.evaluate(skup_validacija_ulaz, skup_validacija_cilj, verbose=2)
print()


izlaz = mreza.predict(skup_validacija_ulaz, verbose=0)
izlaz_oznaka = np.argmax(izlaz, axis=1)
stvarno_oznaka = np.argmax(skup_validacija_cilj, axis=1)


tacnost = accuracy_score(stvarno_oznaka, izlaz_oznaka)
f1_macro = f1_score(stvarno_oznaka, izlaz_oznaka, average="macro")
preciznost_m = precision_score(stvarno_oznaka, izlaz_oznaka,
                               average="macro", zero_division=0)
odziv_m = recall_score(stvarno_oznaka, izlaz_oznaka,
                       average="macro", zero_division=0)

print(f"Тачност (accuracy):    {tacnost:.4f}")
print(f"F1 (macro):            {f1_macro:.4f}   - референтна метрика")
print(f"Прецизност (macro):    {preciznost_m:.4f}")
print(f"Одзив (macro):         {odziv_m:.4f}")
print()


print("Детаљан извештај о класификацији:")
print(classification_report(stvarno_oznaka, izlaz_oznaka,
                            target_names=tezina_nazivi))

matrica_konfuzije = confusion_matrix(stvarno_oznaka, izlaz_oznaka)
print("Матрица конфузије (апсолутне вредности):")
print(matrica_konfuzije)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


im0 = axes[0].imshow(matrica_konfuzije, cmap="Blues")
axes[0].set_title("Матрица конфузије (апсолутно)")
axes[0].set_xlabel("Предвиђена класа")
axes[0].set_ylabel("Стварна класа")
axes[0].set_xticks(range(len(tezina_nazivi)))
axes[0].set_yticks(range(len(tezina_nazivi)))
axes[0].set_xticklabels(tezina_nazivi, rotation=20, ha="right")
axes[0].set_yticklabels(tezina_nazivi)
for i in range(matrica_konfuzije.shape[0]):
    for j in range(matrica_konfuzije.shape[1]):
        axes[0].text(j, i, str(matrica_konfuzije[i, j]), ha="center", va="center",
                     color="black" if matrica_konfuzije[i, j] < matrica_konfuzije.max()/2 else "white",
                     fontsize=11)
plt.colorbar(im0, ax=axes[0])


matrica_konf_norm = matrica_konfuzije.astype("float") / matrica_konfuzije.sum(axis=1)[:, np.newaxis]
im1 = axes[1].imshow(matrica_konf_norm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Матрица конфузије (нормализовано)")
axes[1].set_xlabel("Предвиђена класа")
axes[1].set_ylabel("Стварна класа")
axes[1].set_xticks(range(len(tezina_nazivi)))
axes[1].set_yticks(range(len(tezina_nazivi)))
axes[1].set_xticklabels(tezina_nazivi, rotation=20, ha="right")
axes[1].set_yticklabels(tezina_nazivi)
for i in range(matrica_konf_norm.shape[0]):
    for j in range(matrica_konf_norm.shape[1]):
        axes[1].text(j, i, f"{matrica_konf_norm[i, j]:.2f}",
                     ha="center", va="center",
                     color="black" if matrica_konf_norm[i, j] < 0.5 else "white",
                     fontsize=11)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()


print("\nПогрешне процене (првих 20):")
brojac = 0
for stvarno, procenjeno in zip(stvarno_oznaka, izlaz_oznaka):
    if stvarno != procenjeno:
        print("\t излаз '{}' \n\t\t очекивано '{}'".format(
            mapa_kod_naziv[procenjeno],
            mapa_kod_naziv[stvarno]))
        brojac += 1
        if brojac >= 20:
            break

ukupno_pogresnih = np.sum(stvarno_oznaka != izlaz_oznaka)
print(f"\nУкупно погрешних процена: {ukupno_pogresnih} / {len(stvarno_oznaka)}")
print(f"Проценат погрешних: {100 * ukupno_pogresnih / len(stvarno_oznaka):.2f}%")
