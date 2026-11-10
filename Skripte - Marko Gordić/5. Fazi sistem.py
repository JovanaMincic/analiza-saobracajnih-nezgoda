#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мамданијев фази систем за предвиђање тежине саобраћајне незгоде.

Из обележја незгоде правимо неколико улаза на скали 0-10 који описују колико је
ситуација ризична (рањивост учесника, ризик маневра, време, категорија пута), а
онда фази правилима типа АКО-ОНДА добијамо тежину као број између 0 и 2. Тај број
на крају праговима сврставамо у једну од три класе: мат. штета, повређени,
погинули.

Поступак је уобичајен за фази закључивање: фазификација улаза, примена правила,
агрегација и дефазификација центроидом до једног броја. Механику ради библиотека
skfuzzy, ми дефинишемо улазе, функције припадности и базу правила.

Прагове за претварање броја у класу бирамо на тренинг скупу (максимизујемо
macro-F1), а меримо на одвојеном тест скупу. Поредимо 4 варијанте улаза:
NLP, NLP+време, NLP+пут и NLP+време+пут.

Улаз:  Podaci/CSV/Nezgode 2025 - meteo i opis.csv (+ osm колоне из ...putevi.csv)
Излаз: Podaci/CSV/Fazi metrike.csv, Podaci/CSV/Poređenje pristupa.csv,
       Podaci/Grafici/Matrica konfuzije - *.png
"""

# %% 0. Увоз библиотека

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, precision_recall_fscore_support)

NAZIVI_TEZINE = ['Мат. штета', 'Повређени', 'Погинули']

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
data['osm_kategorija_puta'] = putevi['osm_kategorija_puta'].values
print(f"Учитано: {data.shape[0]} незгода")


# %% 2. Извођење оштрих (crisp) улаза из обележја

# Фази систем очекује нумеричке улазе, а наша обележја су бинарна. Зато из њих
# конструишемо четири лингвистичке променљиве на скали 0-10 које описују
# ризичност ситуације. Бодовање прати рангирање из статистичке анализе
# (скрипта 4): обележја са већом просечном тежином добијају више бодова.

def ulaz_ranjivost(podaci):
    """Рањивост учесника 0-10 (пешак највише, паркирано/животиња најмање).

    Бодови прате просечну тежину незгоде по обележју: пешак ~1.03 -> 9,
    шинско ~0.59 -> 5, а животиња (~0.07) и паркирано возило (~0.07) су
    практично једнаки и при самом дну скале (готово искључиво мат. штета).
    """
    uslovi = [podaci['opis_pesak'] == 1, podaci['opis_parkirano'] == 1,
              podaci['opis_zivotinja'] == 1, podaci['opis_sinsko'] == 1]
    vrednosti = [9.0, 0.5, 1.0, 5.0]
    return np.select(uslovi, vrednosti, default=4.0)


def ulaz_rizik_manevra(podaci):
    """Ризик маневра/судара 0-10 - поредак прати просечну тежину из анализе.

    Бодови су рангирани по просечној тежини незгоде када је маневар присутан
    (скрипта 4): раскрсница (~0.82) и превртање (~0.80) највише, затим чеони
    (~0.78), па скретање (~0.54), силазак/претицање/отварање врата (~0.47-0.52),
    сустизање (~0.44), вожња уназад (~0.33), а паркирано (~0.07) најниже.
    np.select бира ПРВИ испуњен услов, па су услови поређани од најтежег ка
    најлакшем: ако је присутно више маневара, доминира најризичнији.
    """
    uslovi = [
        podaci['opis_raskrsnica'] == 1,
        podaci['opis_prevrtanje'] == 1,
        podaci['opis_ceoni'] == 1,
        podaci['opis_skretanje'] == 1,
        podaci['opis_silazak'] == 1,
        podaci['opis_preticanje'] == 1,
        podaci['opis_otvaranje_vrata'] == 1,
        podaci['opis_sustizanje'] == 1,
        podaci['opis_unazad'] == 1,
        podaci['opis_parkirano'] == 1,
    ]
    vrednosti = [9.0, 9.0, 8.0, 6.0, 5.0, 5.0, 5.0, 4.0, 3.0, 1.0]
    return np.select(uslovi, vrednosti, default=4.0)


# Ризик категорије пута - поредак изведен из статистичке анализе (просечне
# тежине по категорији): магистрале су најризичније, сервисни путеви најмање.
RIZIK_PUTA = {'service': 2, 'brza_saobracajnica': 2, 'autoput': 3, 'drugo': 4,
              'naseljeno': 5, 'lokalni': 6, 'regionalni': 6,
              'neklasifikovano': 7, 'magistrala': 8}


def ulaz_kontekst_puta(podaci):
    return podaci['osm_kategorija_puta'].map(RIZIK_PUTA).fillna(5).astype(float).values


def ulaz_vreme(podaci):
    """Ризик временских услова 0-10 (снег/лед највише, ведро најмање)."""
    uslovi = [podaci['weather_code'].isin([71, 73, 75]) | (podaci['temperature_2m'] < 0),
              podaci['weather_code'].isin([61, 63, 65, 95]),
              podaci['weather_code'].isin([51, 53, 55]) | (podaci['precipitation'] > 0)]
    return np.select(uslovi, [9.0, 6.0, 4.0], default=1.0)


data['ranjivost'] = ulaz_ranjivost(data)
data['rizik_manevra'] = ulaz_rizik_manevra(data)
data['kontekst_puta'] = ulaz_kontekst_puta(data)
data['vreme'] = ulaz_vreme(data)


# %% 3. ФАЗИФИКАЦИЈА: функције припадности + БАЗА ПРАВИЛА

# Универзум дискурса је опсег вредности над којим су термови (фази скупови)
# дефинисани: улази су на скали 0-10, а излаз (тежина) на скали 0-2.
UNIVERZUM_ULAZA = np.arange(0, 10.01, 0.5)
UNIVERZUM_IZLAZA = np.arange(0, 2.01, 0.01)


def fazi_ulaz(naziv):
    """Језичка улазна променљива са три терма (троугаоне функције припадности).
    """
    promenljiva = ctrl.Antecedent(UNIVERZUM_ULAZA, naziv)
    promenljiva['nizak'] = fuzz.trimf(UNIVERZUM_ULAZA, [0, 0, 4])
    promenljiva['srednji'] = fuzz.trimf(UNIVERZUM_ULAZA, [2, 5, 8])
    promenljiva['visok'] = fuzz.trimf(UNIVERZUM_ULAZA, [6, 10, 10])
    return promenljiva


def fazi_izlaz_tezina():
    """Излазна променљива 'tezina' са три терма на скали 0-2.

    defuzzify_method='centroid' значи да фази резултат претварамо у један број
    рачунањем тежишта (центроида) добијене површине.
    """
    tezina = ctrl.Consequent(UNIVERZUM_IZLAZA, 'tezina', defuzzify_method='centroid')
    tezina['materijalna'] = fuzz.trimf(UNIVERZUM_IZLAZA, [0, 0, 1])
    tezina['povredjeni'] = fuzz.trimf(UNIVERZUM_IZLAZA, [0, 1, 2])
    tezina['poginuli'] = fuzz.trimf(UNIVERZUM_IZLAZA, [1, 2, 2])
    return tezina


def napravi_sistem(sa_vremenom, sa_putem):
    """Прави базу правила (АКО-ОНДА) за изабрану варијанту улаза.

    Свако правило везује услов (иза АКО) и последицу (иза ОНДА). Оператор &
    је фази И: снага правила је минимум припадности термова у услову. skfuzzy
    потом скупи доприносе свих правила и дефазификује их у један број.
    """
    ranjivost = fazi_ulaz('ranjivost')
    manevar = fazi_ulaz('rizik_manevra')
    tezina = fazi_izlaz_tezina()

    # Основних 9 правила: пуна табела рањивост x ризик маневра (3x3 комбинације
    # термова), где свакој комбинацији додељујемо терм излазне тежине.
    pravila = [
        ctrl.Rule(ranjivost['nizak'] & manevar['nizak'], tezina['materijalna']),
        ctrl.Rule(ranjivost['nizak'] & manevar['srednji'], tezina['materijalna']),
        ctrl.Rule(ranjivost['nizak'] & manevar['visok'], tezina['povredjeni']),
        ctrl.Rule(ranjivost['srednji'] & manevar['nizak'], tezina['materijalna']),
        ctrl.Rule(ranjivost['srednji'] & manevar['srednji'], tezina['materijalna']),
        ctrl.Rule(ranjivost['srednji'] & manevar['visok'], tezina['povredjeni']),
        ctrl.Rule(ranjivost['visok'] & manevar['nizak'], tezina['povredjeni']),
        ctrl.Rule(ranjivost['visok'] & manevar['srednji'], tezina['povredjeni']),
        ctrl.Rule(ranjivost['visok'] & manevar['visok'], tezina['poginuli']),
    ]

    if sa_vremenom:
        vreme = fazi_ulaz('vreme')
        pravila += [
            ctrl.Rule(vreme['visok'] & manevar['visok'], tezina['poginuli']),
            ctrl.Rule(vreme['visok'] & manevar['srednji'], tezina['povredjeni']),
            ctrl.Rule(vreme['nizak'] & ranjivost['nizak'], tezina['materijalna']),
        ]

    if sa_putem:
        put = fazi_ulaz('kontekst_puta')
        pravila += [
            ctrl.Rule(put['visok'] & manevar['visok'], tezina['poginuli']),
            ctrl.Rule(put['visok'] & ranjivost['visok'], tezina['poginuli']),
            ctrl.Rule(put['visok'] & manevar['srednji'], tezina['povredjeni']),
            ctrl.Rule(put['nizak'] & ranjivost['nizak'], tezina['materijalna']),
            ctrl.Rule(put['nizak'] & manevar['nizak'], tezina['materijalna']),
        ]
    return ctrl.ControlSystem(pravila)


# %% 4. ИНФЕРЕНЦИЈА И ДЕФАЗИФИКАЦИЈА (са мемоизацијом)

def defazifikovani_izlaz(sistem, podaci, ulazi):
    """Враћа дефазификован излаз (број 0-2) за сваки ред.

    compute() за дате улазе одради цео фази поступак и врати један број. Пошто
    има мало различитих комбинација улаза, рачунамо излаз једном по комбинацији
    и кеширамо га (мемоизација) - брже него позивати инференцију за свих 33.000
    редова.
    """
    kombinacije = list(zip(*[podaci[ulaz].values for ulaz in ulazi]))
    simulacija = ctrl.ControlSystemSimulation(sistem)
    kes_izlaza = {}
    for kombinacija in set(kombinacije):
        for naziv, vrednost in zip(ulazi, kombinacija):
            simulacija.input[naziv] = vrednost
        simulacija.compute()
        kes_izlaza[kombinacija] = simulacija.output['tezina']
    return np.array([kes_izlaza[k] for k in kombinacije])


# %% 5. Подела на тренинг/тест + подешавање прагова на тренингу

y = data['tezina_num'].values
idx_trening, idx_test = train_test_split(np.arange(len(data)), test_size=0.30,
                                         random_state=42, stratify=y)


def u_klasu(crisp, donji_prag, gornji_prag):
    """Пресликава дефазификовани (оштар) излаз 0-2 у дискретну класу тежине.

    Дефазификација даје континуалан број (нпр. 1.35)
    """
    return np.where(crisp < donji_prag, 0, np.where(crisp < gornji_prag, 1, 2))


def najbolji_pragovi(crisp, y_stvarno):
    """Grid претрага прагова (donji_prag < gornji_prag) која максимизује macro-F1."""
    najbolji_f1, pragovi = -1, (0.5, 1.3)
    for donji_prag in np.arange(0.30, 1.00, 0.05):
        for gornji_prag in np.arange(1.00, 1.75, 0.05):
            f1 = f1_score(y_stvarno, u_klasu(crisp, donji_prag, gornji_prag),
                          average='macro', zero_division=0)
            if f1 > najbolji_f1:
                najbolji_f1 = f1
                pragovi = (round(donji_prag, 2), round(gornji_prag, 2))
    return pragovi


def evaluiraj(sa_vremenom, sa_putem, naziv):
    ulazi = ['ranjivost', 'rizik_manevra']
    if sa_vremenom:
        ulazi.append('vreme')
    if sa_putem:
        ulazi.append('kontekst_puta')
    sistem = napravi_sistem(sa_vremenom, sa_putem)
    crisp = defazifikovani_izlaz(sistem, data, ulazi)

    donji_prag, gornji_prag = najbolji_pragovi(crisp[idx_trening], y[idx_trening])
    y_pred = u_klasu(crisp[idx_test], donji_prag, gornji_prag)
    y_test = y[idx_test]

    preciznost, odziv, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0)
    metrika = {
        'model': naziv, 'pragovi': f'{donji_prag}/{gornji_prag}',
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_macro': round(float(np.mean(f1)), 4),
        'recall_macro': round(float(np.mean(odziv)), 4),
        'f1_mat': round(f1[0], 4), 'f1_pov': round(f1[1], 4), 'f1_pog': round(f1[2], 4),
        'prec_mat': round(preciznost[0], 4), 'prec_pov': round(preciznost[1], 4),
        'prec_pog': round(preciznost[2], 4),
        'rec_mat': round(odziv[0], 4), 'rec_pov': round(odziv[1], 4),
        'rec_pog': round(odziv[2], 4),
    }
    matrica = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    return metrika, matrica


# %% 6. Покретање 4 варијанте

metrika_nlp, matrica_nlp = evaluiraj(False, False, 'NLP')
metrika_vreme, matrica_vreme = evaluiraj(True, False, 'NLP+vreme')
metrika_put, matrica_put = evaluiraj(False, True, 'NLP+put')
metrika_pun, matrica_pun = evaluiraj(True, True, 'NLP+vreme+put')

metrike = pd.DataFrame([metrika_nlp, metrika_vreme, metrika_put, metrika_pun])
os.makedirs('Podaci/CSV', exist_ok=True)
metrike.to_csv('Podaci/CSV/Fazi metrike.csv', index=False, encoding='utf-8')

print("\nМетрике на тест скупу:")
print(metrike.to_string(index=False))


# %% 7. Матрице конфузије

def nacrtaj_matricu(matrica, naslov, putanja):
    matrica_norm = matrica / matrica.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(matrica_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3), NAZIVI_TEZINE)
    ax.set_yticks(range(3), NAZIVI_TEZINE)
    ax.set_xlabel('Предвиђено')
    ax.set_ylabel('Стварно')
    ax.set_title(naslov)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrica[i, j]}\n{matrica_norm[i, j]*100:.0f}%',
                    ha='center', va='center',
                    color='white' if matrica_norm[i, j] > 0.5 else '#222', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label='Удео по реду')
    plt.tight_layout()
    plt.savefig(putanja)
    plt.show()


nacrtaj_matricu(matrica_nlp, 'Матрица конфузије - Фази (NLP)',
                'Podaci/Grafici/Matrica konfuzije - NLP.png')
nacrtaj_matricu(matrica_vreme, 'Матрица конфузије - Фази (NLP+време)',
                'Podaci/Grafici/Matrica konfuzije - NLP i vreme.png')
nacrtaj_matricu(matrica_put, 'Матрица конфузије - Фази (NLP+пут)',
                'Podaci/Grafici/Matrica konfuzije - NLP i put.png')
nacrtaj_matricu(matrica_pun, 'Матрица конфузије - Фази (NLP+време+пут)',
                'Podaci/Grafici/Matrica konfuzije - pun model.png')


# %% 8. Табела поређења приступа (+ колона за Јованину неуронску мрежу)

# Већински baseline: увек предвиђа најчешћу класу (мат. штета).
# Служи као доња референца због израженог дисбаланса класа.
y_baseline = np.zeros_like(y[idx_test])
baseline = {
    'accuracy': round(accuracy_score(y[idx_test], y_baseline), 4),
    'f1_macro': round(f1_score(y[idx_test], y_baseline, average='macro', zero_division=0), 4),
    'recall_macro': round(recall_score(y[idx_test], y_baseline, average='macro', zero_division=0), 4),
}


def kolona_metrika(metrika):
    return [metrika['accuracy'], metrika['f1_macro'], metrika['recall_macro'],
            metrika['f1_mat'], metrika['f1_pov'], metrika['f1_pog']]


poredjenje = pd.DataFrame({
    'metrika': ['Accuracy', 'F1 (macro)', 'Odziv (macro)',
                'F1 mat.steta', 'F1 povredjeni', 'F1 poginuli'],
    'Vecinski baseline': [baseline['accuracy'], baseline['f1_macro'],
                          baseline['recall_macro'],
                          round(f1_score(y[idx_test], y_baseline,
                                         labels=[0], average=None)[0], 4),
                          0.0, 0.0],
    'NLP': kolona_metrika(metrika_nlp),
    'NLP+vreme': kolona_metrika(metrika_vreme),
    'NLP+put': kolona_metrika(metrika_put),
    'NLP+vreme+put': kolona_metrika(metrika_pun),
    'MLP': ['-'] * 6,
})
poredjenje.to_csv('Podaci/CSV/Poređenje pristupa.csv', index=False, encoding='utf-8')
print("\nПоређење приступа (колона MLP се попуњава резултатима неуронске мреже):")
print(poredjenje.to_string(index=False))
