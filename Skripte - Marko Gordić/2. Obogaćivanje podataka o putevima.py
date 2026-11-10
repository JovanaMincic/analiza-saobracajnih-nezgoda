#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ВАЖНО: Ова скрипта је направљена уз додатну помоћ вештачке интелигенције

Обогаћује незгоде подацима о путној мрежи из локалног OpenStreetMap PBF фајла
(потпуно офлајн геокодирање, без мрежних позива). За сваку незгоду се просторним
спајањем проналази најближи пут, а из тагова се изводе обележја: категорија
пута, брзински лимит, број трака, једносмерност, тунел, мост, подлога и
удаљеност незгоде до пута.

PBF фајл: https://download.geofabrik.de/europe/serbia-latest.osm.pbf
ВАЖНО: обавезно је подесити PUTANJA_PBF испод на локацију PBF фајла.

Улаз:  Podaci/CSV/Nezgode 2025 - meteo.csv (+ OSM PBF)
Излаз: Podaci/CSV/Nezgode 2025 - meteo i putevi.csv
"""

# %% 0. Увоз библиотека

import pandas as pd
import numpy as np
import re
import os
import time
import warnings

import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings('ignore')

# Конфигурација путања и параметара
PUTANJA_PBF = "data/serbia-260522.osm.pbf"
PUTANJA_CSV_IN = "Podaci/CSV/Nezgode 2025 - meteo.csv"
PUTANJA_CSV_OUT = "Podaci/CSV/Nezgode 2025 - meteo i putevi.csv"

CRS_METRICKI = "EPSG:32634"   # UTM зона 34N
CRS_WGS84 = "EPSG:4326"
MAX_RAZDALJINA_M = 150


# %% 1. Провера PBF фајла

print("Валидација PBF фајла:")

if not os.path.exists(PUTANJA_PBF):
    print(f"ГРЕШКА: PBF фајл није пронађен: {PUTANJA_PBF}")
    print("Преузмите га са: https://download.geofabrik.de/europe/serbia-latest.osm.pbf")
    print("Потребно је подесити PUTANJA_PBF у скрипти на праву локацију.")
    raise FileNotFoundError(PUTANJA_PBF)

velicina_pbf_mb = os.path.getsize(PUTANJA_PBF) / 1024 / 1024
print(f"PBF пронађен: {PUTANJA_PBF}")
print(f"Величина: {velicina_pbf_mb:.1f} MB")


# %% 2. Учитавање незгода

data = pd.read_csv(PUTANJA_CSV_IN, encoding='utf-8')
print(f"Димензије података: {data.shape}")

data = data.drop_duplicates().reset_index(drop=True)
print(f"Димензије података након уклањања дупликата: {data.shape}")


# %% 3. Учитавање OSM мреже путева из PBF фајла

# GDAL/OGR OSM драјвер види 5 лејера у PBF фајлу:
# points, lines, multilinestrings, multipolygons, other_relations.
# Путеви се налазе у 'lines' лејеру. Филтрирамо само линије које имају
# 'highway' таг (остало су реке, стазе, ограде итд).

print("Учитавање OSM путева (траје 30s-2min, све се дешава локално)...")

pocetak = time.time()

putevi = gpd.read_file(
    PUTANJA_PBF,
    layer='lines',
    where="highway IS NOT NULL",
    engine='pyogrio'
)

print(f"Учитано за {time.time()-pocetak:.1f}s")
print(f"Број сегмената пута: {len(putevi):,}")
print(f"Доступне колоне: {list(putevi.columns)}")


# %% 4. Парсирање 'other_tags' колоне

# GDAL OSM драјвер пакује већину тагова у колону 'other_tags' у hstore формату:
#   "maxspeed"=>"50","lanes"=>"2","oneway"=>"yes"
# Тагови који су нам потребни се издвајају у засебне колоне.

print("Парсирање тагова:")

pocetak = time.time()


def parsiraj_hstore(tekst):
    if tekst is None or pd.isna(tekst):
        return {}
    return dict(re.findall(r'"([^"]+)"=>"([^"]*)"', str(tekst)))


# Парсирамо све тагове одједном (много брже него таг по таг над сировим текстом)
if 'other_tags' in putevi.columns:
    parsirani_tagovi = putevi['other_tags'].apply(parsiraj_hstore)
    putevi['maxspeed'] = parsirani_tagovi.apply(lambda tagovi: tagovi.get('maxspeed'))
    putevi['lanes']    = parsirani_tagovi.apply(lambda tagovi: tagovi.get('lanes'))
    putevi['oneway']   = parsirani_tagovi.apply(lambda tagovi: tagovi.get('oneway'))
    putevi['surface']  = parsirani_tagovi.apply(lambda tagovi: tagovi.get('surface'))
    putevi['tunnel']   = parsirani_tagovi.apply(lambda tagovi: tagovi.get('tunnel'))
    putevi['bridge']   = parsirani_tagovi.apply(lambda tagovi: tagovi.get('bridge'))
    putevi['ref']      = parsirani_tagovi.apply(lambda tagovi: tagovi.get('ref'))
else:
    for kolona in ['maxspeed', 'lanes', 'oneway', 'surface', 'tunnel', 'bridge', 'ref']:
        putevi[kolona] = None

print(f"Тагови парсирани за {time.time()-pocetak:.1f}s")

# Задржавамо само колоне битне за просторно спајање
zadrzane_kolone = ['highway', 'name', 'ref', 'maxspeed', 'lanes',
                   'oneway', 'surface', 'tunnel', 'bridge', 'geometry']
zadrzane_kolone = [k for k in zadrzane_kolone if k in putevi.columns]
putevi = putevi[zadrzane_kolone].copy()

print("Топ 10 highway тагова:")
print(putevi['highway'].value_counts().head(10).to_string())


# %% 5. Пројекција у метрички систем (UTM 34N)

# Незгоде претварамо у GeoDataFrame, па оба слоја пројектујемо у UTM зону 34N
# како би растојања између тачака и путева била изражена у метрима.

pocetak = time.time()

tacke_gdf = gpd.GeoDataFrame(
    data.copy(),
    geometry=[Point(lon, lat) for lon, lat in zip(data['lon'], data['lat'])],
    crs=CRS_WGS84
)

tacke_proj = tacke_gdf.to_crs(CRS_METRICKI)
putevi_proj = putevi.to_crs(CRS_METRICKI)

print(f"Пројекција готова за {time.time()-pocetak:.1f}s")


# %% 6. Просторно спајање - најближи пут за сваку незгоду

pocetak = time.time()

spojeno = gpd.sjoin_nearest(
    tacke_proj,
    putevi_proj,
    how='left',
    distance_col='osm_udalj_do_puta_m'
)

# Ако једна тачка има више једнако удаљених путева (врло ретко), узимамо први
spojeno = spojeno[~spojeno.index.duplicated(keep='first')]

print(f"Просторно спајање готово за {time.time()-pocetak:.1f}s")
print(f"Упарено: {len(spojeno):,} незгода")

print("Статистика удаљености од најближег пута:")
print(f"  медијана: {spojeno['osm_udalj_do_puta_m'].median():.1f} m")
print(f"  90. перцентил: {spojeno['osm_udalj_do_puta_m'].quantile(0.9):.1f} m")
print(f"  максимум: {spojeno['osm_udalj_do_puta_m'].max():.1f} m")

daleko = spojeno['osm_udalj_do_puta_m'] > MAX_RAZDALJINA_M
print(f"Тачака даље од {MAX_RAZDALJINA_M}m од најближег пута: "
      f"{daleko.sum()} ({100*daleko.mean():.1f}%)")


# %% 7. Парсирање сирових OSM вредности у нумеричке

def parsiraj_brzinski_limit(vrednost):
    """Парсира maxspeed таг у број km/h."""
    if vrednost is None or pd.isna(vrednost):
        return np.nan
    vrednost = str(vrednost).strip().lower()
    if vrednost in ('none', 'walk', 'signals', 'no'):
        return np.nan
    if 'urban' in vrednost: return 50.0
    if 'rural' in vrednost: return 80.0
    if 'motorway' in vrednost: return 130.0
    broj = re.search(r'\d+', vrednost)
    if not broj: return np.nan
    limit = float(broj.group())
    if 'mph' in vrednost: limit = limit * 1.60934
    return limit


def parsiraj_ceo_broj(vrednost):
    """Парсира цео број из стринга."""
    if vrednost is None or pd.isna(vrednost):
        return np.nan
    try:
        return int(str(vrednost).split(';')[0])
    except (ValueError, IndexError):
        return np.nan


def parsiraj_da_ne(vrednost):
    """OSM 'yes'/'no' у 1/0."""
    if vrednost is None or pd.isna(vrednost):
        return 0
    return 1 if str(vrednost).lower() == 'yes' else 0


spojeno['osm_brzinski_limit'] = spojeno['maxspeed'].apply(parsiraj_brzinski_limit)
spojeno['osm_lanes']  = spojeno['lanes'].apply(parsiraj_ceo_broj)
spojeno['osm_oneway'] = spojeno['oneway'].apply(parsiraj_da_ne)
spojeno['osm_tunnel'] = spojeno['tunnel'].apply(parsiraj_da_ne)
spojeno['osm_bridge'] = spojeno['bridge'].apply(parsiraj_da_ne)


# %% 8. Изведена обележја (категорије пута)

def kategorija_puta(highway_tag):
    """Мапира сирови OSM highway таг у семантичку категорију пута."""
    if highway_tag is None or pd.isna(highway_tag):
        return 'nepoznato'
    highway_tag = str(highway_tag).lower()
    if highway_tag in ('motorway', 'motorway_link'):   return 'autoput'
    if highway_tag in ('trunk', 'trunk_link'):          return 'brza_saobracajnica'
    if highway_tag in ('primary', 'primary_link'):      return 'magistrala'
    if highway_tag in ('secondary', 'secondary_link'):  return 'regionalni'
    if highway_tag in ('tertiary', 'tertiary_link'):    return 'lokalni'
    if highway_tag in ('residential', 'living_street'): return 'naseljeno'
    if highway_tag in ('service',):                      return 'service'
    if highway_tag in ('unclassified',):                 return 'neklasifikovano'
    return 'drugo'


print("Креирање изведених обележја...")

spojeno['osm_tip_puta_raw']    = spojeno['highway']
spojeno['osm_kategorija_puta'] = spojeno['highway'].apply(kategorija_puta)
spojeno['osm_ime_puta']        = spojeno['name']
spojeno['osm_ref_puta']        = spojeno['ref']
spojeno['osm_surface']         = spojeno['surface']

spojeno['osm_je_autoput'] = spojeno['osm_kategorija_puta'].isin(
    ['autoput', 'brza_saobracajnica']).astype(int)
spojeno['osm_je_magistrala'] = (
    spojeno['osm_kategorija_puta'] == 'magistrala').astype(int)
spojeno['osm_je_naseljeno'] = spojeno['osm_kategorija_puta'].isin(
    ['naseljeno', 'service']).astype(int)
spojeno['osm_je_evropski_put'] = spojeno['osm_ref_puta'].fillna('').astype(
    str).str.contains(r'^E[ -]?\d', regex=True).astype(int)
spojeno['osm_pouzdano'] = (
    spojeno['osm_udalj_do_puta_m'] <= MAX_RAZDALJINA_M).astype(int)


# %% 9. Враћање OSM колона у оригинални DataFrame

osm_kolone = [
    'osm_tip_puta_raw', 'osm_kategorija_puta', 'osm_brzinski_limit',
    'osm_lanes', 'osm_oneway', 'osm_tunnel', 'osm_bridge',
    'osm_ime_puta', 'osm_ref_puta', 'osm_surface',
    'osm_udalj_do_puta_m', 'osm_je_autoput', 'osm_je_magistrala',
    'osm_je_naseljeno', 'osm_je_evropski_put', 'osm_pouzdano'
]

for kolona in osm_kolone:
    if kolona in spojeno.columns:
        data[kolona] = spojeno[kolona].values
    else:
        data[kolona] = np.nan


# %% 10. Извештај о покривености

ukupno = len(data)


def pokrivenost(kolona, opis):
    nedostaje = data[kolona].isna().sum()
    if data[kolona].dtype == 'object':
        prazno = (data[kolona].fillna('') == '').sum()
        nepoznato = (data[kolona] == 'nepoznato').sum()
        nedostupno = max(nedostaje, prazno, nepoznato)
    else:
        nedostupno = nedostaje
    procenat = 100 * nedostupno / ukupno
    print(f"  {opis:<40} фали за {nedostupno:5} ({procenat:5.1f}%)")


print(f"Укупно редова: {ukupno:,}\n")
pokrivenost('osm_tip_puta_raw',    'Тип пута (сирови OSM таг)')
pokrivenost('osm_kategorija_puta', 'Категорија пута (семантичка)')
pokrivenost('osm_brzinski_limit',  'Брзински лимит (km/h)')
pokrivenost('osm_lanes',           'Број трака')
pokrivenost('osm_ime_puta',        'Име пута')
pokrivenost('osm_ref_puta',        'Референца пута (E-75, M-21...)')
pokrivenost('osm_surface',         'Тип подлоге')

print("\nРасподела osm_kategorija_puta:")
print(data['osm_kategorija_puta'].value_counts().to_string())

# Просечна тежина по категорији пута
mapa_tezine = {'Sa mat.stetom': 0, 'Sa povredjenim': 1, 'Sa poginulim': 2}
if 'tezina_num' not in data.columns:
    data['tezina_num'] = data['tezina'].map(mapa_tezine)

print("\nПросечна тежина по категорији пута:")
po_kategoriji = data.groupby('osm_kategorija_puta')['tezina_num'].agg(
    ['mean', 'count']).round(3).sort_values('mean', ascending=False)
print(po_kategoriji.to_string())

print("\nКључна поређења:")
for naziv, kolona in [('Аутопут', 'osm_je_autoput'),
                      ('Магистрала', 'osm_je_magistrala'),
                      ('Насељено', 'osm_je_naseljeno')]:
    broj = (data[kolona] == 1).sum()
    if broj > 0:
        prosek = data[data[kolona] == 1]['tezina_num'].mean()
        print(f"  {naziv:<12} тежина={prosek:.3f}  n={broj}")
print(f"  {'Глобално':<12} тежина={data['tezina_num'].mean():.3f}  n={len(data)}")


# %% 11. Чување обогаћених података

os.makedirs(os.path.dirname(PUTANJA_CSV_OUT), exist_ok=True)
data.to_csv(PUTANJA_CSV_OUT, index=False, encoding='utf-8')

print(f"\nСачувано: {PUTANJA_CSV_OUT}")
print(f"Додато {sum(1 for kolona in data.columns if kolona.startswith('osm_'))} osm_ колона")
print(f"Укупно колона у излазу: {len(data.columns)}")
