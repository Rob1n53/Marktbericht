#!/usr/bin/env python3
"""
Markt-Bericht für langfristiges Investieren
===========================================

    python main.py           echte Daten laden, Bericht erzeugen
    python main.py --demo    Testlauf mit simulierten Daten (ohne Internet)
    python main.py --open    Bericht danach im Browser öffnen

Erzeugt im Ausgabe-Ordner:
    index.html          vollständiger Bericht mit Diagrammen (Webseite)
    zusammenfassung.md  Kurzfassung, die GitHub als Benachrichtigung per E-Mail verschickt
    titel.txt           Betreff der Benachrichtigung
    senden.txt          nur vorhanden, wenn heute eine Benachrichtigung rausgehen soll
"""

from __future__ import annotations

import argparse
import json
import math
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import analyse as A
import bericht
import daten as D

BASIS = Path(__file__).resolve().parent


def lade_wert(lader, ticker, jahre):
    try:
        return lader(ticker, jahre)
    except Exception as e:  # noqa: BLE001
        print(f"  {ticker}: {e}")
        return None


def seiten_url() -> str:
    """Adresse der GitHub-Webseite, z. B. https://name.github.io/marktbericht/"""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return ""
    besitzer, name = repo.split("/", 1)
    return f"https://{besitzer.lower()}.github.io/{name}/"


def main():
    ap = argparse.ArgumentParser(description="Markt-Bericht für langfristiges Investieren")
    ap.add_argument("--config", default=str(BASIS / "config.json"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--ausgabe", default=str(BASIS / "ausgabe"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    jahre = int(cfg.get("historie_jahre", 15))
    schwelle = float(cfg.get("signal_schwelle", 60))
    kurse = D.demo_kurse if args.demo else D.kurse
    firmendaten = D.demo_firmendaten if args.demo else D.firmendaten
    nachrichten = D.demo_nachrichten if args.demo else D.nachrichten
    jetzt = datetime.now(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)

    # --- Marktumfeld ------------------------------------------------------
    print("Marktumfeld …")
    m = cfg["markt"]
    indizes, umfeld, sektoren = [], [], []
    sp500 = vix = eurusd = None
    for x in m["indizes"]:
        s = lade_wert(kurse, x["ticker"], 2)
        if s is not None:
            indizes.append({**x, "kurs": s.iloc[-1], "perf": A.wertentwicklung(s)})
            if x["ticker"] == "^GSPC":
                sp500 = s
    for x in m["umfeld"]:
        s = lade_wert(kurse, x["ticker"], 2)
        if s is not None:
            umfeld.append({**x, "kurs": s.iloc[-1], "perf": A.wertentwicklung(s)})
            if x["ticker"] == "^VIX":
                vix = float(s.iloc[-1])
            if x["ticker"] == "EURUSD=X":
                eurusd = float(s.iloc[-1])
    for x in m["sektoren"]:
        s = lade_wert(kurse, x["ticker"], 1)
        if s is not None:
            sektoren.append({**x, "perf": A.wertentwicklung(s)})
    breite = None
    rsp, spy = lade_wert(kurse, "RSP", 1), lade_wert(kurse, "SPY", 1)
    if rsp is not None and spy is not None and len(rsp) > 63 and len(spy) > 63:
        breite = (rsp.iloc[-1] / rsp.iloc[-64]) / (spy.iloc[-1] / spy.iloc[-64]) - 1
    klima = A.marktklima(vix, sp500, breite)

    # --- ETFs --------------------------------------------------------------
    print("ETFs …")
    etfs = []
    for x in cfg["etfs"]:
        e = {**x}
        try:
            s = kurse(x["ticker"], jahre)
            if len(s) < 300:
                raise ValueError("zu wenig Kurshistorie")
            df = A.kennzahlen(s, "etf")
            e.update(df=df, perf=A.wertentwicklung(s), rb=A.rueckblick(df, schwelle),
                     score=float(df["kurs_score"].iloc[-1]))
            print(f"  {x['ticker']:10s} Score {e['score']:5.1f}")
        except Exception as ex:  # noqa: BLE001
            e["fehler"] = str(ex)
            print(f"  {x['ticker']:10s} FEHLER {ex}")
        etfs.append(e)

    # --- Aktien ------------------------------------------------------------
    print("Aktien …")
    aktien, termine_firmen = [], []
    for x in cfg["aktien"]:
        e = {**x}
        try:
            s = kurse(x["ticker"], jahre)
            df = A.kennzahlen(s, "aktie")
            firma = firmendaten(x["ticker"])
            bew, gruende = A.bewertungs_punkte(firma, float(s.iloc[-1]))
            e.update(df=df, perf=A.wertentwicklung(s), rb=A.rueckblick(df, schwelle), firma=firma,
                     bewertung=bew, gruende=gruende,
                     score=A.gesamt_score_aktie(float(df["kurs_score"].iloc[-1]), bew),
                     kurs_eur=float(s.iloc[-1]) / eurusd if eurusd else float("nan"))
            t = firma.get("zahlen_termin")
            if t is not None and not pd.isna(t):
                termine_firmen.append({"datum": pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t),
                                       "titel": f"Quartalszahlen {x['name']}"})
            print(f"  {x['ticker']:10s} Score {e['score']:5.1f}")
        except Exception as ex:  # noqa: BLE001
            e["fehler"] = str(ex)
            print(f"  {x['ticker']:10s} FEHLER {ex}")
        aktien.append(e)

    # --- Termine & Nachrichten --------------------------------------------
    heute = pd.Timestamp(jetzt.date())
    termine = [{"datum": pd.Timestamp(t["datum"]), "titel": t["titel"]} for t in cfg.get("termine", [])]
    termine = sorted([t for t in termine + termine_firmen
                      if heute <= t["datum"] <= heute + pd.Timedelta(days=30)], key=lambda t: t["datum"])
    print("Nachrichten …")
    n = cfg.get("nachrichten", {})
    news = nachrichten(n.get("feeds", []), int(n.get("anzahl", 8)))

    # --- Bericht -----------------------------------------------------------
    d = {"datum": jetzt, "demo": args.demo, "modus": cfg.get("modus", "taeglich"), "schwelle": schwelle,
         "url": cfg.get("bericht_url") or seiten_url(), "klima": klima, "indizes": indizes, "umfeld": umfeld,
         "sektoren": sektoren, "etfs": etfs, "aktien": aktien, "termine": termine, "nachrichten": news}
    aus = Path(args.ausgabe)
    aus.mkdir(parents=True, exist_ok=True)
    web = aus / "index.html"
    web.write_text(bericht.erstelle(d, web=True), encoding="utf-8")
    print(f"\nBericht: {web}")

    # --- Kurzfassung für die Benachrichtigung ------------------------------
    signale = sum(1 for e in etfs + aktien if not e.get("fehler") and e["score"] >= schwelle)
    ampel = klima["titel"].split(" –")[0]
    art = "Wochenbericht" if d["modus"] == "woechentlich" else "Tagesbericht"
    titel = f"Markt-{art} {jetzt:%d.%m.%Y}: Ampel {ampel} · {signale} Signal{'e' if signale != 1 else ''}"
    (aus / "titel.txt").write_text(titel, encoding="utf-8")
    (aus / "zusammenfassung.md").write_text(bericht.zusammenfassung(d), encoding="utf-8")
    senden = aus / "senden.txt"
    if d["modus"] == "woechentlich" and jetzt.weekday() != int(cfg.get("wochentag_fuer_wochenbericht", 4)):
        senden.unlink(missing_ok=True)
        print("Wochenmodus: heute keine Benachrichtigung.")
    else:
        senden.write_text("ja", encoding="utf-8")

    if args.open:
        webbrowser.open(web.resolve().as_uri())


if __name__ == "__main__":
    main()
