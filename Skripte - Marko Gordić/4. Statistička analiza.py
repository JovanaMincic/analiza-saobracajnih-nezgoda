#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ова скрипта се користи за статистичку анализу везе НЛП обележја (и категорије
пута) са тежином незгоде.

За свако обележје рачунамо три комплементарне мере:
  - Chi-square тест независности - да ли веза обележја и тежине уопште постоји
    (статистичка значајност,p-вредност)
  - Крамеров V - КОЛИКО је веза јака (вредност у опсегу 0-1)
  - Спирманов rho - у ком СМЕРУ обележје помера тежину (само за ординална/
    бинарна обележја, где поредак има смисла)

Резултат је ранг-табела која служи као основа за дефинисање фази правила
(скрипта 5): обележја са јаком и значајном везом постају улази фази система.

Улаз:  Podaci/CSV/Nezgode 2025 - meteo i opis.csv (+ osm колоне из ...putevi.csv)
Излаз: Podaci/CSV/Statistička analiza - rezultati.csv + графикони
"""

# %% 0. Увоз библиотека

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, spearmanr

NAZIVI_TEZINE = ['Мат. штета', 'Повређени', 'Погинули']
BOJE_TEZINE = ['#4C9F70', '#F4A261', '#C0392B']

# Читљиви називи обележја за осе графикона (уместо сирових назива колона).
NAZIVI_OBELEZJA = {
    'opis_tema': 'НЛП тема', 'opis_pesak': 'пешак',
    'opis_parkirano': 'паркирано возило', 'opis_raskrsnica': 'раскрсница',
    'opis_van_raskrsnice': 'ван раскрснице', 'opis_skretanje': 'скретање',
    'opis_pesak_zdesna': 'пешак здесна', 'opis_pesak_sleva': 'пешак слева',
    'opis_u_raskrsnici': 'у раскрсници', 'opis_prepreka': 'препрека',
    'opis_pesak_duz_kolovoza': 'пешак дуж коловоза', 'opis_ceoni': 'чеони судар',
    'opis_zivotinja': 'животиња', 'opis_pesak_stoji': 'пешак стоји',
    'opis_skret_levo': 'скретање лево', 'opis_silazak': 'силазак с пута',
    'opis_trotoar': 'тротоар', 'opis_prevrtanje': 'превртање',
    'opis_isti_smer': 'исти смер', 'opis_supr_smer': 'супротан смер',
    'opis_krivina': 'кривина', 'opis_pravac': 'права деоница',
    'opis_uporedna': 'упоредна вожња', 'opis_unazad': 'вожња уназад',
    'opis_polukruzno': 'полукружно окретање', 'opis_skret_desno': 'скретање десно',
    'opis_sustizanje': 'сустизање', 'opis_preticanje': 'претицање',
    'opis_ukljucivanje': 'укључивање', 'opis_radovi': 'радови на путу',
    'opis_otvaranje_vrata': 'отварање врата', 'kategorija': 'категорија незгоде',
    'osm_kategorija_puta': 'категорија пута', 'mraz': 'мраз', 'kisa': 'киша',
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})


# %% 1. Учитавање података (опис + osm колоне)

data = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo i opis.csv", encoding='utf-8')
putevi = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo i putevi.csv", encoding='utf-8')

# Оба фајла су редно поравната (исти извор, исти редослед) па osm колоне
# додајемо директно по позицији.
for kolona in [k for k in putevi.columns if k.startswith('osm_')]:
    data[kolona] = putevi[kolona].values

print(f"Учитано: {data.shape[0]} незгода")
print("Расподела тежине:",
      data['tezina_num'].value_counts(normalize=True).round(3).sort_index().to_dict())


# %% 2. Помоћне функције

def kramerov_v(x, y):
    """Крамеров V - јачина асоцијације две категоричке променљиве, опсег 0-1.

    Формула: V = sqrt( chi2 / (n * min(r-1, c-1)) ), где су r и c број врста и
    колона табеле контингенције, а n укупан број узорака. За разлику од сировог
    chi2 (који расте са n), V је нормализован на 0-1 па је упоредив међу
    обележјима. Уобичајена интерпретација: V < 0.1 занемарљива веза, око 0.3
    умерена, > 0.5 јака.
    """
    tabela = pd.crosstab(x, y)
    chi2 = chi2_contingency(tabela)[0]
    n = tabela.sum().sum()
    min_dimenzija = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    return float(np.sqrt(chi2 / (n * min_dimenzija))) if min_dimenzija else 0.0


def analiza_para(obelezje, tezina, naziv, ordinalno):
    """Враћа речник са chi2, p-вредношћу, Крамеровим V и Спирмановим rho.

    Chi-square тест независности проверава да ли се расподела тежине разликује
    у зависности од обележја (мала p-вредност = веза постоји, није случајна).
    Спирманов rho рачунамо само када је ordinalno=True (бинарна/ординална
    обележја), јер он мери монотону везу преко рангова и захтева природан
    поредак вредности; за чисто категоричка обележја поредак не постоји па су
    rho и смер недефинисани (np.nan).
    """
    tabela = pd.crosstab(obelezje, tezina)
    chi2, p_vrednost, stepeni_slobode, _ = chi2_contingency(tabela)
    rho = spearmanr(obelezje, tezina)[0] if ordinalno else np.nan
    return {
        'obelezje': naziv,
        'chi2': round(chi2, 1),
        'stepeni_slobode': stepeni_slobode,
        'p_vrednost': p_vrednost,
        'kramerov_v': round(kramerov_v(obelezje, tezina), 4),
        'spearman_rho': round(rho, 4) if ordinalno else np.nan,
        'smer': ('+' if rho > 0 else '-') if ordinalno else '',
        # значајно = веза је статистички поуздана уз уобичајени праг p < 0.05
        # (мање од 5% шансе да је овакав образац настао случајно)
        'znacajno': 'DA' if p_vrednost < 0.05 else '-',
    }


# %% 3. Бинарна opis_ обележја наспрам тежине

tezina = data['tezina_num']
opis_kolone = [k for k in data.columns if k.startswith('opis_') and k != 'opis_tema']

rezultati = []
for kolona in opis_kolone:
    # праг од 30 позитивних случајева обезбеђује довољну статистичку
    # поузданост: испод тога су ћелије табеле контингенције преретке и
    # chi-square тест постаје непоуздан, па такво обележје прескачемо
    if data[kolona].sum() < 30:
        continue
    rezultati.append({**analiza_para(data[kolona], tezina, kolona, ordinalno=True),
                      'broj_pozitivnih': int(data[kolona].sum())})


# %% 4. Категоричка обележја наспрам тежине

# Код категоричких обележја нема природног поретка па Спирманов rho и смер
# нису дефинисани - рачунамо само Chi-square и Крамеров V.
for naziv in ['kategorija', 'opis_tema', 'osm_kategorija_puta']:
    rezultati.append({**analiza_para(data[naziv], tezina, naziv, ordinalno=False),
                      'broj_pozitivnih': data[naziv].nunique()})


# %% 5. Провера метео сигнала (очекивано слаб)

data['kisa'] = (data['precipitation'] > 0).astype(int)
data['mraz'] = (data['temperature_2m'] < 0).astype(int)
for kolona in ['kisa', 'mraz']:
    rezultati.append({**analiza_para(data[kolona], tezina, kolona, ordinalno=True),
                      'broj_pozitivnih': int(data[kolona].sum())})


# %% 6. Обједињена ранг-табела + чување

stat = (pd.DataFrame(rezultati)
        .sort_values('kramerov_v', ascending=False)
        .reset_index(drop=True))

redosled_kolona = ['obelezje', 'broj_pozitivnih', 'chi2', 'stepeni_slobode',
                   'p_vrednost', 'kramerov_v', 'spearman_rho', 'smer', 'znacajno']
stat = stat[redosled_kolona]

os.makedirs('Podaci/CSV', exist_ok=True)
stat.to_csv('Podaci/CSV/Statistička analiza - rezultati.csv', index=False, encoding='utf-8')

print("Ранг-табела (сортирано по Крамеровом V):")
print(stat.drop(columns='p_vrednost').to_string(index=False))


# %% 7. Графикон - Крамеров V по обележју

znacajna_obelezja = stat[stat['znacajno'] == 'DA'].sort_values('kramerov_v')
nazivi = znacajna_obelezja['obelezje'].map(NAZIVI_OBELEZJA).fillna(
    znacajna_obelezja['obelezje'])

fig, ax = plt.subplots(figsize=(10, 9))
boje = ['#C0392B' if smer == '+' else '#2980B9' if smer == '-' else '#7F8C8D'
        for smer in znacajna_obelezja['smer']]
ax.barh(nazivi, znacajna_obelezja['kramerov_v'],
        color=boje, edgecolor='white', linewidth=0.5)
for i, vrednost in enumerate(znacajna_obelezja['kramerov_v']):
    ax.text(vrednost + 0.003, i, f'{vrednost:.3f}', va='center', fontsize=8.5)
ax.set_xlabel('Крамеров V')
ax.set_ylabel('Обележје')
ax.set_title('Јачина везе обележја са тежином незгоде')
ax.set_xlim(0, znacajna_obelezja['kramerov_v'].max() * 1.12)
ax.legend(handles=[Patch(facecolor='#C0392B', label='повећава тежину'),
                   Patch(facecolor='#2980B9', label='смањује тежину'),
                   Patch(facecolor='#7F8C8D', label='без смера')],
          loc='lower right')
plt.tight_layout()
plt.savefig('Podaci/Grafici/Kramerov V.png')
plt.show()


# %% 8. Графикон - расподела тежине за кључна обележја

kljucna_obelezja = ['opis_pesak', 'opis_ceoni', 'opis_silazak', 'opis_skretanje',
                    'opis_sustizanje', 'opis_parkirano']
kljucna_obelezja = [k for k in kljucna_obelezja if k in data.columns]

redovi = []
for obelezje in kljucna_obelezja:
    raspodela = (data.loc[data[obelezje] == 1, 'tezina_num']
                 .value_counts(normalize=True).reindex([0, 1, 2]).fillna(0) * 100)
    redovi.append(raspodela.values)
raspodele = pd.DataFrame(redovi, index=kljucna_obelezja, columns=NAZIVI_TEZINE)
raspodele = raspodele.loc[raspodele[NAZIVI_TEZINE[1:]].sum(axis=1).sort_values().index]
raspodele.index = [NAZIVI_OBELEZJA.get(k, k) for k in raspodele.index]

fig, ax = plt.subplots(figsize=(11, 6))
dno = np.zeros(len(raspodele))
for kolona, boja in zip(raspodele.columns, BOJE_TEZINE):
    ax.bar(raspodele.index, raspodele[kolona], bottom=dno, label=kolona,
           color=boja, edgecolor='white', linewidth=0.8)
    for i, vrednost in enumerate(raspodele[kolona]):
        if vrednost >= 4:
            ax.text(i, dno[i] + vrednost / 2, f'{vrednost:.0f}%',
                    ha='center', va='center', color='white',
                    fontsize=9, fontweight='bold')
    dno += raspodele[kolona].values
ax.set_ylabel('Проценат незгода (%)')
ax.set_xlabel('Обележје')
ax.set_title('Расподела тежине када је обележје присутно')
ax.legend(title='Тежина', loc='upper left', bbox_to_anchor=(1.01, 1))
ax.set_ylim(0, 100)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('Podaci/Grafici/Crosstab ključnih obeležja.png')
plt.show()
