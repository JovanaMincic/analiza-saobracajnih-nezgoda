#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ова скрипта се користила за истражну анализу података.
"""

# %% 0. Библиотеке

import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})

BOJE = ["steelblue", "orange", "crimson"]
TEZINA_RED = {"Sa mat.stetom": 0, "Sa povredjenim": 1, "Sa poginulim": 2}
TEZINA_NAZIVI = ["Sa mat.stetom", "Sa povredjenim", "Sa poginulim"]


# %% 1. Пресловљавање латинице у ћирилицу

# Филтрирање и груписање остају на латиници (вредности из података), а само
# текст који иде на графиконе пресловљавамо у ћирилицу овом функцијом.
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


# %% 2. Учитавање и изведена временска обележја

data = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo.csv", encoding="utf-8")

data["datum_vreme"] = pd.to_datetime(data["datum_vreme"],
                                     format="%d.%m.%Y,%H:%M", errors="coerce")
data["sat"] = data["datum_vreme"].dt.hour
data["mesec"] = data["datum_vreme"].dt.month
data["dan_u_nedelji"] = data["datum_vreme"].dt.dayofweek
godisnje = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
data["godisnje_doba"] = data["mesec"].map(godisnje)

# Ординално кодирање тежине по озбиљности исхода (0 < 1 < 2). Прави се ЈЕДНОМ
# и користи само у нумеричким анализама (просечна тежина, Spearman, MI,
# корелациона матрица). За све описне графиконе користимо текстуалну 'tezina'.
data["tezina_kod"] = data["tezina"].map(TEZINA_RED)

print("Облик скупа:", data.shape)
print(data.dtypes)


# %% 3. Квалитет података

print("\nНедостајуће вредности по колони:")
print(data.isna().sum())
print("\nБрој дупликата:", data.duplicated().sum())
print(f"Распон датума: {data['datum_vreme'].min()} до {data['datum_vreme'].max()}")
print("\nОсновна статистика нумеричких обележја:")
print(data[["temperature_2m", "precipitation", "wind_speed_10m",
            "cloud_cover", "weather_code"]].describe().round(2))


# %% 4. Расподела циљног обележја (тежина)

raspodela = data["tezina"].value_counts()
print("\nРасподела тежине:")
print(raspodela)
print("\nПроцентуално:")
print((raspodela / len(data) * 100).round(2))

raspodela_cir = raspodela.rename(index=cir)

fig, ax = plt.subplots(figsize=(8, 5))
raspodela_cir.plot(kind="bar", ax=ax, color=BOJE, edgecolor="black")
ax.set_title("Расподела тежине исхода")
ax.set_xlabel("Тежина исхода")
ax.set_ylabel("Број незгода")
ax.tick_params(axis="x", rotation=20)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=12)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - raspodela težine.png", dpi=120,
            bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
raspodela_cir.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                   colors=BOJE, startangle=140)
ax.set_title("Процентуална расподела тежине")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - procentualna raspodela težine.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 5. Расподела категорија незгоде

raspodela_kat = data["kategorija"].value_counts()
print("\nРасподела категорија незгода:")
print(raspodela_kat)

raspodela_kat_cir = raspodela_kat.rename(index=cir)

fig, ax = plt.subplots(figsize=(11, 5))
raspodela_kat_cir.plot(kind="barh", ax=ax, color="steelblue",
                       edgecolor="black")
ax.set_title("Расподела категорија саобраћајних незгода")
ax.set_xlabel("Број незгода")
ax.set_ylabel("Категорија незгоде")
ax.invert_yaxis()
for p in ax.patches:
    ax.annotate(f"{int(p.get_width()):,}",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="left", va="center", fontsize=11)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - raspodela kategorija.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 6. Веза између категорије и тежине незгоде

# Crosstab над текстуалним вредностима (колоне се сортирају абецедно:
# [Sa mat.stetom, Sa poginulim, Sa povredjenim] -> боје [steelblue, crimson, orange]).
tab_apsolutno = pd.crosstab(data["kategorija"], data["tezina"])
tab_procenat = pd.crosstab(data["kategorija"], data["tezina"],
                           normalize="index") * 100
print("\nТежина по категорији (апсолутно):")
print(tab_apsolutno)
print("\nТежина по категорији (%):")
print(tab_procenat.round(1))

boje_stack = ["steelblue", "crimson", "orange"]
tab_proc_cir = tab_procenat.rename(index=cir, columns=cir)

fig, ax = plt.subplots(figsize=(12, 6))
tab_proc_cir.plot(kind="barh", stacked=True, ax=ax, color=boje_stack,
                  edgecolor="black")
ax.set_title("Тежина исхода по категорији незгоде (%)")
ax.set_xlabel("Проценат (%)")
ax.set_ylabel("Категорија незгоде")
ax.set_xlim(0, 100)
ax.legend(title="Тежина", loc="lower right")
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - težina po kategoriji (procentualno).png",
            dpi=120, bbox_inches="tight")
plt.show()

# Просечна тежина по категорији (користи ординални tezina_kod)
prosek_kat = data.groupby("kategorija")["tezina_kod"].mean().sort_values()
prosek_kat_cir = prosek_kat.rename(index=cir)

fig, ax = plt.subplots(figsize=(11, 5))
prosek_kat_cir.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Просечна тежина по категорији незгоде (0 = мат. штета, 2 = погинули)")
ax.set_xlabel("Просечна тежина")
ax.set_ylabel("Категорија незгоде")
for p in ax.patches:
    ax.annotate(f"{p.get_width():.2f}",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="left", va="center", fontsize=11)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - kategorije i težina.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 7. Метеоролошки услови

meteo = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]
nazivi = ["Температура [°C]", "Падавине [mm]", "Брзина ветра [km/h]",
          "Облачност [%]"]
datoteke = ["EDA - meteo temperatura.png", "EDA - meteo padavine.png",
            "EDA - meteo brzina vetra.png", "EDA - meteo oblačnost.png"]
for kol, naz, dat in zip(meteo, nazivi, datoteke):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data[kol], bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(data[kol].mean(), color="red", linestyle="--",
               label=f"Средња: {data[kol].mean():.2f}")
    ax.set_title(f"Расподела: {naz}")
    ax.set_xlabel(naz)
    ax.set_ylabel("Број незгода")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Podaci/Grafici/{dat}", dpi=120, bbox_inches="tight")
    plt.show()

udeo_suvo = (data["precipitation"] == 0).mean() * 100
print(f"\nУдео незгода без падавина: {udeo_suvo:.1f}%")
print(f"Број различитих WMO кодова времена: {data['weather_code'].nunique()}")

# Метео услови наспрам тежине: групне статистике и образац кише (Марко)
print("\nГрупне статистике метео обележја по тежини:")
print(data.groupby("tezina")[meteo].mean().round(2))
data["ima_kisu"] = (data["precipitation"] > 0).astype(int)
print("\nРасподела тежине по присуству кише (%):")
print((pd.crosstab(data["ima_kisu"], data["tezina"],
                   normalize="index") * 100).round(2))


# %% 8. Временска анализа

mes_naz = ["Јан", "Феб", "Мар", "Апр", "Мај", "Јун", "Јул", "Авг", "Сеп",
           "Окт", "Нов", "Дец"]
dani_naz = ["Пон", "Уто", "Сре", "Чет", "Пет", "Суб", "Нед"]

# Број незгода по месецу
po_mesecu = data.groupby("mesec").size()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(po_mesecu.index, po_mesecu.values, color="steelblue", edgecolor="black")
ax.set_title("Број незгода по месецу")
ax.set_xlabel("Месец")
ax.set_ylabel("Број незгода")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(mes_naz)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - broj nezgoda po mesecu.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Број незгода по сату
po_satu = data.groupby("sat").size()
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(po_satu.index, po_satu.values, marker="o", color="steelblue",
        linewidth=2)
ax.fill_between(po_satu.index, po_satu.values, alpha=0.2, color="steelblue")
ax.set_title("Број незгода по сату у току дана")
ax.set_xlabel("Сат у дану")
ax.set_ylabel("Број незгода")
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - broj nezgoda po satu.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Просечна тежина по месецу
fig, ax = plt.subplots(figsize=(9, 5))
data.groupby("mesec")["tezina_kod"].mean().plot(
    kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Просечна тежина по месецу")
ax.set_xlabel("Месец")
ax.set_ylabel("Просечна тежина")
ax.set_xticklabels(mes_naz, rotation=0)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - težina po mesecu.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Просечна тежина по сату
tez_po_satu = data.groupby("sat")["tezina_kod"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tez_po_satu.index, tez_po_satu.values, marker="o", color="steelblue")
ax.fill_between(tez_po_satu.index, tez_po_satu.values, alpha=0.2,
                color="steelblue")
ax.set_title("Просечна тежина по сату")
ax.set_xlabel("Сат у дану")
ax.set_ylabel("Просечна тежина")
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - težina po satu.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Тежина по дану у недељи (наслагано, број незгода)
tab_dani = pd.crosstab(data["dan_u_nedelji"], data["tezina"])
tab_dani.index = dani_naz
tab_dani_cir = tab_dani.rename(columns=cir)
fig, ax = plt.subplots(figsize=(11, 5))
tab_dani_cir.plot(kind="bar", stacked=True, ax=ax, edgecolor="black",
                  color=["steelblue", "crimson", "orange"])
ax.set_title("Тежина незгоде по дану у недељи")
ax.set_xlabel("Дан у недељи")
ax.set_ylabel("Број незгода")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Тежина", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - težina po danu u nedelji.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Проценат погинулих по дану у недељи
proc_pog = data.groupby("dan_u_nedelji")["tezina"].apply(
    lambda x: (x == "Sa poginulim").mean() * 100)
proc_pog.index = dani_naz
print("\nПроценат погинулих по дану (%):")
print(proc_pog.round(3))
fig, ax = plt.subplots(figsize=(10, 5))
proc_pog.plot(kind="bar", ax=ax, color="crimson", edgecolor="black")
ax.set_title("Проценат смртних исхода по дану у недељи")
ax.set_xlabel("Дан у недељи")
ax.set_ylabel("Проценат погинулих (%)")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - procenat poginulih po danu.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 9. Просторна анализа

udeo_bg = data["pu"].astype(str).str.contains("BEOGRAD", case=False).mean() * 100
print(f"\nУдео незгода у Београду (по ПУ): {udeo_bg:.1f}%")

# Топ 10 општина по броју незгода
top10 = data["opstina"].value_counts().head(10)
print("\nТоп 10 општина по броју незгода:")
print(top10)
top10_cir = top10.rename(index=cir)
fig, ax = plt.subplots(figsize=(10, 5))
top10_cir.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Топ 10 општина по броју незгода")
ax.set_xlabel("Број незгода")
ax.set_ylabel("Општина")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - top opstine.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Просторна расподела по тежини
boje_t = dict(zip(TEZINA_NAZIVI, BOJE))
fig, ax = plt.subplots(figsize=(11, 8))
for tez, boja in boje_t.items():
    pod = data[data["tezina"] == tez]
    ax.scatter(pod["lon"], pod["lat"], c=boja, label=cir(tez), s=2, alpha=0.3)
ax.set_title("Просторна расподела незгода по тежини")
ax.set_xlabel("Географска дужина")
ax.set_ylabel("Географска ширина")
ax.legend(markerscale=5, title="Тежина исхода")
plt.tight_layout()
plt.savefig("Podaci/Grafici/EDA - prostorna raspodela.png", dpi=120,
            bbox_inches="tight")
plt.show()
