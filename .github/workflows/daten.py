"""Daten laden: Kurse, Firmenkennzahlen, Nachrichten. Mit Demo-Modus zum Testen ohne Internet."""

from __future__ import annotations

import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Echte Daten (Yahoo Finance über yfinance)
# ----------------------------------------------------------------------------

def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as e:
        raise SystemExit("yfinance fehlt. Bitte installieren:  pip install -r requirements.txt") from e


def kurse(ticker: str, jahre: int) -> pd.Series:
    """Tägliche Schlusskurse (dividendenbereinigt). Drei Versuche, falls Yahoo kurz hakt."""
    yf = _yf()
    letzter_fehler = None
    for versuch in range(3):
        try:
            df = yf.download(ticker, period=f"{jahre}y", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().astype(float)
                close.index = pd.to_datetime(close.index).tz_localize(None)
                return close
            letzter_fehler = "keine Daten erhalten"
        except Exception as e:  # noqa: BLE001
            letzter_fehler = str(e)
        time.sleep(2 * (versuch + 1))
    raise ValueError(f"{ticker}: {letzter_fehler}")


def firmendaten(ticker: str) -> dict:
    """Bewertung, Analystenziel, Wachstum und nächster Quartalszahlen-Termin."""
    yf = _yf()
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:  # noqa: BLE001
        pass
    termin = None
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if isinstance(d, (list, tuple)) and d:
                termin = pd.Timestamp(d[0])
            elif d is not None:
                termin = pd.Timestamp(d)
    except Exception:  # noqa: BLE001
        pass
    return {
        "kgv": info.get("trailingPE"),
        "kgv_erwartet": info.get("forwardPE"),
        "kursziel": info.get("targetMeanPrice"),
        "empfehlung": info.get("recommendationKey"),
        "umsatzwachstum": info.get("revenueGrowth"),
        "gewinnwachstum": info.get("earningsGrowth"),
        "marktwert": info.get("marketCap"),
        "sektor": info.get("sector"),
        "zahlen_termin": termin,
    }


def nachrichten(feeds: list[dict], anzahl: int) -> list[dict]:
    """Schlagzeilen aus RSS-Feeds, neueste zuerst, Dubletten entfernt."""
    items = []
    for feed in feeds:
        try:
            req = urllib.request.Request(feed["url"], headers={"User-Agent": "Mozilla/5.0 Marktbericht"})
            with urllib.request.urlopen(req, timeout=15) as r:
                root = ET.fromstring(r.read())
            for it in root.iter("item"):
                titel = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                datum = it.findtext("pubDate")
                try:
                    dt = parsedate_to_datetime(datum).replace(tzinfo=None) if datum else None
                except Exception:  # noqa: BLE001
                    dt = None
                if titel:
                    items.append({"titel": titel, "link": link, "quelle": feed["name"], "zeit": dt})
        except Exception as e:  # noqa: BLE001
            print(f"  Nachrichten von {feed['name']} nicht erreichbar: {e}")
    gesehen, ergebnis = set(), []
    for it in sorted(items, key=lambda x: x["zeit"] or datetime.min, reverse=True):
        key = it["titel"].lower()[:60]
        if key not in gesehen:
            gesehen.add(key)
            ergebnis.append(it)
    return ergebnis[:anzahl]


# ----------------------------------------------------------------------------
# Demo-Daten (simuliert, nur zum Ausprobieren)
# ----------------------------------------------------------------------------

def _seed(ticker):
    return sum(ord(c) * (i + 1) for i, c in enumerate(ticker))


def demo_kurse(ticker: str, jahre: int) -> pd.Series:
    rng = np.random.default_rng(_seed(ticker))
    n = jahre * TRADING_DAYS
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    basis = {"^VIX": 18, "^TNX": 4.2, "EURUSD=X": 1.12, "GC=F": 3700, "BZ=F": 85}
    if ticker in basis:  # schwankt um einen Mittelwert
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.98 * x[i - 1] + rng.normal(0, 0.03)
        return pd.Series(basis[ticker] * np.exp(x), index=idx)
    vol = 0.30 if "." not in ticker and "^" not in ticker and len(ticker) <= 5 else 0.17
    r = rng.normal(0.08 / TRADING_DAYS, vol / math.sqrt(TRADING_DAYS), n)
    for _ in range(3 if n > 700 else 0):
        s = rng.integers(200, n - 300)
        r[s:s + rng.integers(40, 160)] -= rng.uniform(0.002, 0.004)
    r[-15:] -= 0.004  # aktueller Rücksetzer
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def demo_firmendaten(ticker: str) -> dict:
    rng = np.random.default_rng(_seed(ticker))
    return {
        "kgv": float(rng.uniform(20, 45)), "kgv_erwartet": float(rng.uniform(18, 35)),
        "kursziel": None, "empfehlung": "buy", "umsatzwachstum": float(rng.uniform(0.02, 0.4)),
        "gewinnwachstum": float(rng.uniform(-0.05, 0.5)), "marktwert": float(rng.uniform(2e12, 5e12)),
        "sektor": "Technology", "zahlen_termin": pd.Timestamp.today().normalize() + pd.Timedelta(days=int(rng.integers(5, 80))),
    }


def demo_nachrichten(feeds, anzahl):
    return [{"titel": f"Beispiel-Schlagzeile {i + 1} (Demo-Modus)", "link": "", "quelle": "Demo",
             "zeit": datetime.now()} for i in range(min(anzahl, 4))]
