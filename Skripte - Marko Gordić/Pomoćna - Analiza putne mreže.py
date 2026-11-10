#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Овде рачунам ранг категорија пута по просечној тежини незгоде (магистрала
најтежа, сервисни пут најлакши). Тај ранг користим у фази систему (скрипта 5)
као речник RIZIK_PUTA, тј. улаз 'kontekst_puta', и он оправдава "+пут" варијанту.

Користим само категорију пута зато што Крамеров V показује да је она далеко
најјача веза са тежином, док су брзински лимит, број трака и специјални типови
пута слаби, па их не узимам као фази улазе. Остало из ранијих верзија (имена
путева, хитмапови по добу дана) било је само истраживачко и није ушло у модел.

Улаз: Podaci/CSV/Nezgode 2025 - meteo i putevi.csv (osm_ колоне)
"""

# %% 0. Библиотеке

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import spearmanr, chi2_contingency

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})


# %% 1. Учитавање (само поуздано геокодиране незгоде)

data = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo i putevi.csv", encoding="utf-8")
data["tezina_num"] = data["tezina"].map(
    {"Sa mat.stetom": 0, "Sa povredjenim": 1, "Sa poginulim": 2})

# osm_pouzdano = 1 значи да је незгода довољно близу пута да је OSM податак валидан.
pouzdani = data[data["osm_pouzdano"] == 1].copy()
GLOBALNI_PROSEK = pouzdani["tezina_num"].mean()
print(f"Поуздано геокодирано: {len(pouzdani):,} од {len(data):,} незгода")
print(f"Глобална просечна тежина: {GLOBALNI_PROSEK:.3f}")


# %% 2. Ранг категорија пута по просечној тежини (улаз RIZIK_PUTA у скрипти 5)

stat_kat = (pouzdani.groupby("osm_kategorija_puta")["tezina_num"]
            .agg(n="count", prosek="mean")
            .sort_values("prosek", ascending=False))
print("\nПросечна тежина по категорији пута:")
print(stat_kat.round(3).to_string())

NAZIVI_KAT = {"autoput": "аутопут", "brza_saobracajnica": "брза саобраћајница",
              "magistrala": "магистрала", "regionalni": "регионални",
              "lokalni": "локални", "neklasifikovano": "некласификовано",
              "naseljeno": "насељено", "service": "сервисни", "drugo": "друго"}
prosek_prikaz = stat_kat["prosek"].rename(index=NAZIVI_KAT)

fig, ax = plt.subplots(figsize=(10, 6))
boje = ["crimson" if v > GLOBALNI_PROSEK else "steelblue"
        for v in stat_kat["prosek"]]
ax.barh(prosek_prikaz.index, prosek_prikaz.values, color=boje, edgecolor="black")
for i, (broj, prosek) in enumerate(zip(stat_kat["n"], stat_kat["prosek"])):
    ax.text(prosek + 0.005, i, f"n={int(broj)}", va="center", fontsize=9)
ax.axvline(GLOBALNI_PROSEK, color="black", linestyle="--", alpha=0.6,
           label=f"глобални просек = {GLOBALNI_PROSEK:.3f}")
ax.set_xlabel("Просечна тежина (0 = мат. штета, 2 = погинули)")
ax.set_ylabel("Категорија пута")
ax.set_title("Просечна тежина незгоде по категорији пута")
ax.invert_yaxis()
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Podaci/Grafici/Fazi - rizik kategorije puta.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 3. Провера: категорија пута је најјачи путни сигнал (зато само њу користим)

def kramerov_v(x, y):
    tabela = pd.crosstab(x, y)
    chi2 = chi2_contingency(tabela)[0]
    n = tabela.to_numpy().sum()
    min_dimenzija = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    return float(np.sqrt(chi2 / (n * min_dimenzija))) if min_dimenzija else 0.0


v_kategorija = kramerov_v(pouzdani["osm_kategorija_puta"], pouzdani["tezina"])
print(f"\nКрамеров V (категорија пута и тежина): {v_kategorija:.3f}")
print("Спирманово ро осталих путних обележја (очекивано слабо, зато не улазе):")
for kolona in ["osm_brzinski_limit", "osm_lanes", "osm_je_autoput",
               "osm_je_magistrala", "osm_je_naseljeno"]:
    if kolona in pouzdani.columns:
        validni = pouzdani[kolona].notna()
        if pouzdani.loc[validni, kolona].nunique() >= 2:
            rho = spearmanr(pouzdani.loc[validni, kolona],
                            pouzdani.loc[validni, "tezina_num"])[0]
            print(f"  {kolona:<22} ро = {rho:+.3f}")
