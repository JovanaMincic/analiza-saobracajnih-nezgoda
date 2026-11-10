#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обрада природног језика над колоном 'opis' (категоричка, 68 јединствених
вредности). Из описа се изводе бинарна обележја:
  - врста учесника незгоде
  - тип маневра
  - тип локације
  - врста судара
  - смер кретања приликом незгоде

Улаз:  Podaci/CSV/Nezgode 2025 - meteo.csv
Излаз: Podaci/CSV/Nezgode 2025 - meteo i opis.csv
"""

# %% 0. Увоз библиотека

import os
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})


# %% 1. Учитавање скупа података

data = pd.read_csv("Podaci/CSV/Nezgode 2025 - meteo.csv", encoding='utf-8')
print(f"Димензије података: {data.shape}")

data = data.drop_duplicates().reset_index(drop=True)
print(f"Димензије података након уклањања дупликата: {data.shape}")

# Проверавамо колону опис која нам је јако битна
print(f"Број јединствених вредности унутар колоне 'opis': {data['opis'].nunique()}")
print(f"NaN унутар колоне 'opis': {data['opis'].isna().sum()}")


# %% 2. Дефинишемо речник НЛП шаблона

# Користимо стемове речи унутар шаблона како би успешно детектовали присутност
# истих унутар текста.

NLP_FEATURES = {
    # Могуће врсте учесника
    'opis_pesak':           r'pešak|peša',
    'opis_parkirano':       r'parkiran',
    'opis_zivotinja':       r'životinj',
    'opis_sinsko':          r'šinsk|tramvaj',

    # Могући типови маневра
    'opis_skretanje':       r'skretanj|skreće',
    'opis_skret_levo':      r'skretanj.{0,15}ulevo|ulevo.{0,15}skret',
    'opis_skret_desno':     r'skretanj.{0,15}udesno|udesno.{0,15}skret',
    'opis_polukruzno':      r'polukružn',
    'opis_preticanje':      r'preticanj',
    'opis_sustizanje':      r'sustizanj',
    'opis_unazad':          r'unazad',
    'opis_ukljucivanje':    r'uključivanj',
    'opis_uporedna':        r'uporedn',
    'opis_ceoni':           r'čeon',

    # Могуће локације
    'opis_raskrsnica':      r'raskrsnic',
    'opis_van_raskrsnice':  r'van raskrsnic',
    'opis_u_raskrsnici':    r'(?<!van )u raskrsnic|petlj',
    'opis_krivina':         r'krivin',
    'opis_pravac':          r'na pravcu',
    'opis_trotoar':         r'trotoar|biciklističk',
    'opis_radovi':          r'radov',
    'opis_prepreka':        r'prepreka',

    # Могући типови судара
    'opis_silazak':         r'silazak',
    'opis_prevrtanje':      r'prevrtanj',
    'opis_otvaranje_vrata': r'otvaranj.{0,20}vrata',
    'opis_isti_smer':       r'isti smer|istom smeru',
    'opis_supr_smer':       r'suprotn.{0,5}smer',

    # Могућ смер пешака
    'opis_pesak_sleva':     r'pešak.{0,30}sleva|sleva.{0,30}pešak',
    'opis_pesak_zdesna':    r'pešak.{0,30}zdesna|zdesna.{0,30}pešak',
    'opis_pesak_stoji':     r'pešak stoji',
    'opis_pesak_duz_kolovoza': r'duž kolovoza',
}

print(f"Успешно дефинисано {len(NLP_FEATURES)} НЛП feature-a")


# %% 3. Примена шаблона на податке

# нормализујемо текст да избегнемо НаН описе (у нашем скупу их нема) и користимо
# функцију lower како капитализација текста не би утицала на накнадне провере
opis_lc = data['opis'].fillna('').str.lower()

# Креирамо пбинарне колоне. За сваки шаблон постављамо вредност 1 уколико се
# појам појављује у тексту, уколико не онда 0.
for ime, pat in NLP_FEATURES.items():
    data[ime] = opis_lc.str.contains(pat, regex=True, na=False).astype(int)

print(f"Успешно додато {len(NLP_FEATURES)} бинарних колона у скуп података!")


# %% 4. Тематска категоризација

# Свака незгода се мапира на тачно једну од 9 могућих група
# користимо следећу хијерарију услед могућег преклапања у неким ситуацијама
def odredi_temu(row):
    if row['opis_pesak'] == 1:
        return 'pesak'
    if row['opis_parkirano'] == 1:
        return 'parkirano'
    if row['opis_zivotinja'] == 1 or row['opis_prepreka'] == 1:
        return 'objekat_prepreka'
    if row['opis_silazak'] == 1 or row['opis_prevrtanje'] == 1:
        return 'silazak'
    if row['opis_ceoni'] == 1:
        return 'ceoni'
    if row['opis_raskrsnica'] == 1:
        return 'raskrsnica'
    if row['opis_skretanje'] == 1:
        return 'skretanje'
    if row['opis_sustizanje'] == 1 or row['opis_preticanje'] == 1:
        return 'sustizanje_preticanje'
    if row['opis_unazad'] == 1:
        return 'unazad'
    return 'ostalo'


data['opis_tema'] = data.apply(odredi_temu, axis=1)


# %% 5. Pokrivenost svakog feature-a

# Кодирамо тежине у бројчане вредности.

# У питању је ординална скала, односно, категорије имају природан поредак
# али размаци између њих нису једнаки или дефинисани.
mapa_tezine = {'Sa mat.stetom': 0, 'Sa povredjenim': 1, 'Sa poginulim': 2}
if 'tezina_num' not in data.columns:
    data['tezina_num'] = data['tezina'].map(mapa_tezine)

rezultati = []
for ime_feature in NLP_FEATURES.keys():
    broj_pozitivnih = int(data[ime_feature].sum())
    # број 30 користимо као границу испод које сматрамо да нема довољно
    # података да би закључак био поуздан
    if broj_pozitivnih < 30:
        continue

    # Користимо Spearman-а а не Pearson-а зато што Pearson мери линеарну везу
    # и користи саме нумеричке вредности. Улед ординалне скале, ми би могли променити
    # кодирање на 0/1/10 што би и даље било валидно али би тест давао другачије
    # резултате.
    spearman_rho, p_vrednost = spearmanr(data[ime_feature], data['tezina_num'])
    
    prosek_sa_pojmom = data[data[ime_feature] == 1]['tezina_num'].mean()
    prosek_bez_pojma = data[data[ime_feature] == 0]['tezina_num'].mean()
    razlika_proseka = prosek_sa_pojmom - prosek_bez_pojma

    rezultati.append({
        'pojam': ime_feature,
        'broj_pozitivnih': broj_pozitivnih,
        'procenat': round(100 * broj_pozitivnih / len(data), 1),
        'prosek_sa_pojmom': round(prosek_sa_pojmom, 3),
        'prosek_bez_pojma': round(prosek_bez_pojma, 3),
        'razlika_proseka': round(razlika_proseka, 3),
        'spearman_rho': round(spearman_rho, 4),
        'p_vrednost': p_vrednost,
        'znacajno': 'DA' if p_vrednost < 0.05 else '-'
    })

nlp_df = pd.DataFrame(rezultati).sort_values(
    'razlika_proseka', key=lambda kolona: kolona.abs(), ascending=False)

print("Приказ појмова рангираних по разлици у просечној тежини:")
print(nlp_df.drop(columns='p_vrednost').to_string(index=False))


# %% 6. Извештај о тематским групама

print("Тематске групе незгода:")

tema_stats = data.groupby('opis_tema').agg(
    broj_nezgoda=('tezina_num', 'count'),
    procenat=('tezina_num', lambda grupa: round(100*len(grupa)/len(data), 1)),
    prosecna_tezina=('tezina_num', 'mean')
).round(3).sort_values('prosecna_tezina', ascending=False)

print(tema_stats.to_string())

print(f"\nУкупно класификованих незгода: {len(data)}")
print(f"Група 'ostalo' (није покривена другим темама): "
      f"{(data['opis_tema'] == 'ostalo').sum()} "
      f"({100*(data['opis_tema'] == 'ostalo').mean():.1f}%)")


# %% 7. Чување обогаћених података

izlaz = "Podaci/CSV/Nezgode 2025 - meteo i opis.csv"
data.to_csv(izlaz, index=False, encoding='utf-8')

print(f"Сачувано: {izlaz}")
print(f"Оригиналних колона: 14")
print(f"Нових opis колона: {sum(1 for kolona in data.columns if kolona.startswith('opis_'))}")
print(f"Укупно колона у излазу: {len(data.columns)}")
print(f"Број редова: {len(data)}")


# %% 8. Подешавања за графиконе

os.makedirs('Podaci/Grafici', exist_ok=True)

BOJE_TEZINE = ['#4C9F70', '#F4A261', '#C0392B']
BOJA_POZ = '#C0392B'
BOJA_NEG = '#2980B9'

# Читљиви називи обележја за осе графикона (уместо сирових назива колона).
NAZIVI_OBELEZJA = {
    'opis_pesak': 'пешак', 'opis_parkirano': 'паркирано возило',
    'opis_zivotinja': 'животиња', 'opis_sinsko': 'шинско возило',
    'opis_skretanje': 'скретање', 'opis_skret_levo': 'скретање лево',
    'opis_skret_desno': 'скретање десно', 'opis_polukruzno': 'полукружно окретање',
    'opis_preticanje': 'претицање', 'opis_sustizanje': 'сустизање',
    'opis_unazad': 'вожња уназад', 'opis_ukljucivanje': 'укључивање',
    'opis_uporedna': 'упоредна вожња', 'opis_ceoni': 'чеони судар',
    'opis_raskrsnica': 'раскрсница', 'opis_van_raskrsnice': 'ван раскрснице',
    'opis_u_raskrsnici': 'у раскрсници', 'opis_krivina': 'кривина',
    'opis_pravac': 'права деоница', 'opis_trotoar': 'тротоар',
    'opis_radovi': 'радови на путу', 'opis_prepreka': 'препрека',
    'opis_silazak': 'силазак с пута', 'opis_prevrtanje': 'превртање',
    'opis_otvaranje_vrata': 'отварање врата', 'opis_isti_smer': 'исти смер',
    'opis_supr_smer': 'супротан смер', 'opis_pesak_sleva': 'пешак слева',
    'opis_pesak_zdesna': 'пешак здесна', 'opis_pesak_stoji': 'пешак стоји',
    'opis_pesak_duz_kolovoza': 'пешак дуж коловоза',
}


# %% 9. Фреквенција НЛП појмова

freq_df = nlp_df.sort_values('broj_pozitivnih')
freq_naz = freq_df['pojam'].map(NAZIVI_OBELEZJA).fillna(freq_df['pojam'])

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(freq_naz, freq_df['broj_pozitivnih'], color='#3F88C5')
ax.bar_label(bars, labels=[f'{broj}  ({procenat}%)' for broj, procenat
                           in zip(freq_df['broj_pozitivnih'], freq_df['procenat'])],
             padding=3, fontsize=8.5)
ax.set_xlabel('Број незгода у којима је обележје присутно')
ax.set_ylabel('Обележје')
ax.set_title(f'Фреквенција НЛП обележја (n = {len(data)})')
ax.margins(x=0.15)
plt.tight_layout()
plt.savefig('Podaci/Grafici/NLP - frekvencija obeležja.png')
plt.show()


# %% 10. Разлика у просечној тежини (са појмом наспрам без појма)

diff_df = nlp_df.sort_values('razlika_proseka')
diff_naz = diff_df['pojam'].map(NAZIVI_OBELEZJA).fillna(diff_df['pojam'])
boje = [BOJA_POZ if razlika > 0 else BOJA_NEG for razlika in diff_df['razlika_proseka']]
oznake = [f'{razlika:+.3f}{" *" if znacajno == "DA" else ""}'
          for razlika, znacajno in zip(diff_df['razlika_proseka'], diff_df['znacajno'])]

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(diff_naz, diff_df['razlika_proseka'], color=boje)
ax.bar_label(bars, labels=oznake, padding=3, fontsize=8.5)
ax.axvline(0, color='black', linewidth=0.9)
ax.set_xlabel('Разлика просечне тежине: са обележјем - без обележја (* = p < 0.05)')
ax.set_ylabel('Обележје')
ax.set_title('Утицај НЛП обележја на тежину незгоде')
ax.legend(handles=[Patch(facecolor=BOJA_POZ, label='повећава тежину'),
                   Patch(facecolor=BOJA_NEG, label='смањује тежину')],
          loc='lower right')
ax.margins(x=0.18)
plt.tight_layout()
plt.savefig('Podaci/Grafici/NLP - razlika u težini.png')
plt.show()
