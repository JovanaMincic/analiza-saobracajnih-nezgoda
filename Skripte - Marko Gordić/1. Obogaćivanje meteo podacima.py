#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ВАЖНО: Ова скрипта је направљена уз додатну помоћ вештачке интелигенције

Обогаћује изворни скуп незгода временским подацима из Open-Meteo API-ja. За
сваку незгоду добавља:
  - temperature_2m (°C): температура
  - precipitation (mm): падавине
  - wind_speed_10m (km/h): брзина ветра
  - cloud_cover (%): покривеност неба облацима
  - weather_code (WMO код: 0=ведро, 45=магла, 61-65=киша, 71-75=снег, 95=грмљавина)

Локације су заокружене на једну децималу да се смањи број позива услед употребе
бесплатног API-ja.

Улаз:  Podaci/CSV/Nezgode 2025.csv
Излаз: Podaci/CSV/Nezgode 2025 - meteo.csv
"""

import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ULAZNI_CSV = Path("Podaci/CSV/Nezgode 2025.csv")
IZLAZNI_CSV = Path("Podaci/CSV/Nezgode 2025 - meteo.csv")

SATNE_PROMENLJIVE = "temperature_2m,precipitation,wind_speed_10m,cloud_cover,weather_code"
API_URL = "https://archive-api.open-meteo.com/v1/archive"

SIRINA_TRAKE = 40


# Ова функција је написана уз помоћ вештачке интелигенције са циљем праћења напредка
def traka_napretka(trenutno, ukupno, oznaka="", greske=0):
    procenat = trenutno / ukupno if ukupno else 0
    popunjeno = int(SIRINA_TRAKE * procenat)
    traka = "#" * popunjeno + "-" * (SIRINA_TRAKE - popunjeno)
    tekst_gresaka = f" | {greske} грешака" if greske else ""
    sys.stdout.write(f"\r  [{traka}] {trenutno}/{ukupno} ({procenat:.0%}) {oznaka}{tekst_gresaka}")
    sys.stdout.flush()
    if trenutno == ukupno:
        print()


ZAGLAVLJE = [
    "id", "pu", "opstina", "datum_vreme",
    "lon", "lat", "tezina", "kategorija", "opis",
    "temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover", "weather_code",
]


def parsiraj_datum(tekst_datuma):
    """'05.01.2025' -> '2025-01-05'"""
    dan, mesec, godina = tekst_datuma.strip().split(".")
    return f"{godina}-{mesec}-{dan}"


def parsiraj_sat(tekst_vremena):
    """'18:50' -> 18"""
    return int(tekst_vremena.strip().split(":")[0])


def preuzmi_meteo_za_mesec(lat, lon, datum_od, datum_do, broj_pokusaja=3):
    """Позива Open-Meteo API за један месечни опсег уз понављање при грешци."""
    parametri = (
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={datum_od}&end_date={datum_do}"
        f"&hourly={SATNE_PROMENLJIVE}"
        f"&timezone=Europe/Belgrade"
    )
    url = API_URL + parametri

    for pokusaj in range(broj_pokusaja):
        try:
            zahtev = urllib.request.Request(url)
            with urllib.request.urlopen(zahtev, timeout=30) as odgovor:
                podaci = json.loads(odgovor.read().decode())

            po_satima = podaci["hourly"]
            vremenske_tacke = po_satima["time"]
            promenljive = SATNE_PROMENLJIVE.split(",")

            rezultat = {}
            for i, tacka in enumerate(vremenske_tacke):
                datum = tacka[:10]
                sat = int(tacka[11:13])
                if datum not in rezultat:
                    rezultat[datum] = {}
                rezultat[datum][sat] = {p: po_satima[p][i] for p in promenljive}

            return rezultat
        except urllib.error.HTTPError as greska:
            if greska.code == 429 and pokusaj < broj_pokusaja - 1:
                cekanje = 30 * (pokusaj + 1)
                print(f"    Превише захтева (429), чекам {cekanje}s...")
                time.sleep(cekanje)
            else:
                raise

    return {}


def meseci_sa_podacima(datumi):
    """Из скупа датума враћа месечне опсеге (само месеце који имају незгоде)."""
    from datetime import date as dt_date, timedelta
    skup_meseci = set()
    for datum in datumi:
        parsiran = dt_date.fromisoformat(datum)
        skup_meseci.add((parsiran.year, parsiran.month))

    opsezi = []
    for godina, mesec in sorted(skup_meseci):
        pocetak = dt_date(godina, mesec, 1)
        kraj = (pocetak.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        opsezi.append((pocetak.isoformat(), kraj.isoformat()))
    return opsezi


def preuzmi_meteo(lat, lon, datumi):
    """Преузима временске податке само за месеце у којима је било незгода."""
    rezultat = {}
    for datum_od, datum_do in meseci_sa_podacima(datumi):
        mesec = preuzmi_meteo_za_mesec(lat, lon, datum_od, datum_do)
        for datum, sati in mesec.items():
            rezultat[datum] = sati
        time.sleep(0.15)
    return rezultat


def main():
    # 1. Учитавамо све редове из изворног CSV фајла
    print("Учитавам CSV...")
    redovi = []
    with open(ULAZNI_CSV, newline="", encoding="utf-8") as f:
        citac = csv.reader(f)
        for red in citac:
            redovi.append(red)
    print(f"  Укупно редова: {len(redovi)}")

    # 2. Групишемо незгоде по заокруженој локацији и скупљамо датуме за сваку
    print("Групишем локације...")
    datumi_po_lokaciji = defaultdict(set)  # (lat, lon) -> скуп датума
    for red in redovi:
        lon_zaokruzen = round(float(red[4]), 1)
        lat_zaokruzen = round(float(red[5]), 1)
        datum = parsiraj_datum(red[3].split(",")[0])
        datumi_po_lokaciji[(lat_zaokruzen, lon_zaokruzen)].add(datum)

    # Пребројавамо укупан број АПИ позива (месечних опсега)
    ukupno_poziva = 0
    for datumi in datumi_po_lokaciji.values():
        skup_meseci = set()
        for datum in datumi:
            delovi = datum.split("-")
            skup_meseci.add((delovi[0], delovi[1]))
        ukupno_poziva += len(skup_meseci)

    print(f"  Јединствених локација: {len(datumi_po_lokaciji)}")
    print(f"  Укупно АПИ позива (месечних): {ukupno_poziva}")

    # 3. Позивамо АПИ за сваку локацију
    print("Преузимам временске податке...")
    kes_meteo = {}
    ukupno = len(datumi_po_lokaciji)
    broj_gresaka = 0

    for i, ((lat_zaokruzen, lon_zaokruzen), datumi) in enumerate(datumi_po_lokaciji.items()):
        kljuc = (lat_zaokruzen, lon_zaokruzen)
        try:
            kes_meteo[kljuc] = preuzmi_meteo(lat_zaokruzen, lon_zaokruzen, datumi)
        except Exception as greska:
            print(f"\n  ГРЕШКА за ({lat_zaokruzen}, {lon_zaokruzen}): {greska}")
            kes_meteo[kljuc] = {}
            broj_gresaka += 1

        traka_napretka(i + 1, ukupno, oznaka="локација", greske=broj_gresaka)

    # 4. Спајамо метео податке са незгодама и чувамо излазни CSV
    print("Спајам податке и чувам...")
    with open(IZLAZNI_CSV, "w", newline="", encoding="utf-8") as f:
        pisac = csv.writer(f)
        pisac.writerow(ZAGLAVLJE)

        upareno = 0
        for red in redovi:
            lon_zaokruzen = round(float(red[4]), 1)
            lat_zaokruzen = round(float(red[5]), 1)
            datum = parsiraj_datum(red[3].split(",")[0])
            sat = parsiraj_sat(red[3].split(",")[1])

            meteo = kes_meteo.get((lat_zaokruzen, lon_zaokruzen), {}).get(datum, {}).get(sat, {})

            meteo_kolone = [
                meteo.get("temperature_2m", ""),
                meteo.get("precipitation", ""),
                meteo.get("wind_speed_10m", ""),
                meteo.get("cloud_cover", ""),
                meteo.get("weather_code", ""),
            ]

            if meteo:
                upareno += 1

            pisac.writerow(red + meteo_kolone)

    print("\nГотово!")
    print(f"  Упарено: {upareno}/{len(redovi)} ({100*upareno/len(redovi):.1f}%)")
    print(f"  Сачувано у: {IZLAZNI_CSV}")


if __name__ == "__main__":
    main()
