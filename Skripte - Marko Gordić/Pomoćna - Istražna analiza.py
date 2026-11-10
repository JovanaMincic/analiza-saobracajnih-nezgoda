#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Општа дескриптивна анализа је у "Zajednička istražna analiza.py". Овде остаје
само оно што је конкретно обликовало фази систем (скрипта 5).

Три ствари које одавде користим:
  1. Класе су јако неуравнотежене (погинули око 1.4%), па користим F1 уместо
     тачности и стратификовану поделу на тренинг и тест.
  2. Време је слаб и немонотон предиктор (просечна тежина је највиша по ведром
     времену и благо опада ка лошијем, све близу глобалног просека), па је
     "време" само груб, секундаран фази улаз.
  3. Контекст локације јако мења тежину (расте од урбаних ка ризичним
     локацијама), што мотивише "+пут" варијанту.

Бодовање улаза 'ranjivost' и 'rizik_manevra' долази из НЛП обележја (скрипта 3)
и статистичке анализе (скрипта 4), а 'kontekst_puta' из анализе путне мреже.
Ова скрипта само оправдава дизајн.

Улаз: Podaci/CSV/Nezgode 2025 - meteo.csv
"""

# %% 0. Библиотеке

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import spearmanr

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


# %% 1. Учитавање и припрема обележја

data = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo.csv", encoding="utf-8")
data["datum_vreme"] = pd.to_datetime(data["datum_vreme"],
                                     format="%d.%m.%Y,%H:%M", errors="coerce")
data["tezina_num"] = data["tezina"].map(
    {"Sa mat.stetom": 0, "Sa povredjenim": 1, "Sa poginulim": 2})


# Композит "лоше време" 0-7: киша + облачност + ветар + мраз (мраз се дупло бодује).
def kisa_skala(padavine):
    return (0 if padavine == 0 else 1 if padavine < 0.5 else 2 if padavine < 2.5
            else 3 if padavine < 7.5 else 4)


def oblacnost_skala(oblacnost):
    return 0 if oblacnost <= 25 else 1 if oblacnost <= 75 else 2


def vetar_skala(vetar):
    return 0 if vetar < 5 else 1 if vetar < 15 else 2 if vetar < 25 else 3


data["vreme_lose"] = (data["precipitation"].apply(kisa_skala)
                      + data["cloud_cover"].apply(oblacnost_skala)
                      + data["wind_speed_10m"].apply(vetar_skala)
                      + (data["temperature_2m"] < 0).astype(int) * 2)

# Ризик локације: општине (>=100 незгода) у 3 групе по просечној тежини.
opstine = data.groupby("opstina")["tezina_num"].agg(["mean", "count"])
opstine = opstine[opstine["count"] >= 100]
opstine["rizik"] = pd.qcut(opstine["mean"], 3, labels=["низак", "средњи", "висок"])
data["rizik_lokacije"] = data["opstina"].map(opstine["rizik"]).astype("object")


# %% 2. Класна неравнотежа (F1 + стратификација)

print("Расподела класа (%):")
print((data["tezina"].value_counts(normalize=True) * 100).round(2))
print("погинули ~1.4%: користим F1 (не тачност) и стратификовану поделу.\n")


# %% 3. Време: слаб и немонотон сигнал (секундаран фази улаз)

rho = spearmanr(data["precipitation"], data["tezina_num"])[0]
print(f"Спирманово ро (падавине и тежина) глобално: {rho:+.3f}  (готово нула)")

u_kriva = data.groupby("vreme_lose")["tezina_num"].agg(["mean", "count"])
u_kriva = u_kriva[u_kriva["count"] >= 50]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(u_kriva.index, u_kriva["mean"], "o-", color="steelblue", linewidth=2,
        markersize=8)
ax.axhline(data["tezina_num"].mean(), color="black", linestyle="--", alpha=0.6,
           label=f"глобални просек = {data['tezina_num'].mean():.3f}")
ax.set_xlabel("Композит лошег времена (0 = идеално, 7 = најлошије)")
ax.set_ylabel("Просечна тежина")
ax.set_title("Просечна тежина незгоде по композиту лошег времена")
ax.set_xticks(u_kriva.index)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("Podaci/Grafici/Fazi - U kriva vremena.png", dpi=120,
            bbox_inches="tight")
plt.show()


# %% 4. Контекст јако стратификује тежину (контекстуални улаз +пут)

gradijent = (data.groupby("rizik_lokacije", observed=True)["tezina_num"].mean()
             .reindex(["низак", "средњи", "висок"]))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(gradijent.index, gradijent.values,
       color=["steelblue", "orange", "crimson"], edgecolor="black")
for i, vrednost in enumerate(gradijent.values):
    ax.text(i, vrednost + 0.01, f"{vrednost:.3f}", ha="center", va="bottom")
ax.set_xlabel("Ризик локације (општине по просечној тежини)")
ax.set_ylabel("Просечна тежина")
ax.set_ylim(0, gradijent.max() * 1.15)
ax.set_title("Контекст локације јако мења тежину")
plt.tight_layout()
plt.savefig("Podaci/Grafici/Fazi - rizik lokacije.png", dpi=120,
            bbox_inches="tight")
plt.show()

# Симпсонов образац: ефекат времена готово нестаје глобално, али се види у контексту.
maska_visok = data["rizik_lokacije"] == "висок"
rho_kontekst = spearmanr(data.loc[maska_visok, "precipitation"],
                         data.loc[maska_visok, "tezina_num"])[0]
print(f"Спирманово ро (падавине - тежина): глобално {rho:+.3f}, "
      f"у ризичним локацијама {rho_kontekst:+.3f}")
