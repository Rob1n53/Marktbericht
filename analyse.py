"""Kennzahlen, Einstiegs-Score, Marktklima und Rückblick-Test (Backtest)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Kurs-Bausteine des Scores: (Gewicht, ab welchem Wert volle Punkte)
#   ETFs schwanken weniger als Einzelaktien, deshalb strengere Grenzen.
KURS_BAUSTEINE = {
    "etf":   {"rueckgang": (30, 0.20), "linie200": (20, 0.10), "rsi": (20, 30), "trend": (20, 2.0), "stabil": (10, None)},
    "aktie": {"rueckgang": (30, 0.35), "linie200": (20, 0.15), "rsi": (20, 30), "trend": (20, 2.0), "stabil": (10, None)},
}
BAUSTEIN_NAMEN = {
    "rueckgang": "Abstand zum Hoch", "linie200": "200-Tage-Linie", "rsi": "Wochen-RSI",
    "trend": "5-Jahres-Trend", "stabil": "Stabilisierung", "bewertung": "Bewertung",
}

STUFEN = [
    (70, "Sehr guter Einstiegsmoment", "#1a7f37"),
    (50, "Guter Einstiegsmoment", "#4c9a2a"),
    (25, "Neutral – Sparplan normal weiter", "#6e7781"),
    (0, "Eher teuer – keine Extra-Käufe", "#bc4c00"),
]


def stufe(score: float):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "keine Bewertung", "#6e7781"
    for grenze, text, farbe in STUFEN:
        if score >= grenze:
            return text, farbe
    return STUFEN[-1][1], STUFEN[-1][2]


# ----------------------------------------------------------------------------
# Grund-Indikatoren
# ----------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def trend_abweichung(close: pd.Series, fenster: int = 5 * TRADING_DAYS) -> pd.Series:
    """Abstand des Kurses zu seiner eigenen 5-Jahres-Trendlinie (in Standardabweichungen).
    Verwendet an jedem Tag nur Daten bis zu diesem Tag."""
    fenster = min(fenster, max(TRADING_DAYS, len(close) // 2))
    y = np.log(close.values)
    x_c = np.arange(fenster, dtype=float) - (fenster - 1) / 2
    sxx = (x_c ** 2).sum()
    out = np.full(len(y), np.nan)
    for i in range(fenster - 1, len(y)):
        yw = y[i - fenster + 1:i + 1]
        m = yw.mean()
        resid = yw - (m + (x_c * (yw - m)).sum() / sxx * x_c)
        sd = resid.std()
        out[i] = resid[-1] / sd if sd > 0 else 0.0
    return pd.Series(out, index=close.index)


def _skala(wert, schlecht, gut):
    return np.clip((wert - schlecht) / (gut - schlecht), 0, 1)


def kennzahlen(close: pd.Series, art: str = "etf") -> pd.DataFrame:
    df = pd.DataFrame({"kurs": close})
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["hoch"] = close.cummax()
    df["rueckgang"] = close / df["hoch"] - 1
    df["abstand200"] = close / df["sma200"] - 1
    wochen = close.resample("W-FRI").last().dropna()
    df["rsi_w"] = rsi(wochen).reindex(df.index, method="ffill")
    df["trend_z"] = trend_abweichung(close)
    df["vola"] = np.log(close).diff().rolling(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS)

    b = KURS_BAUSTEINE[art]
    teile = {
        "rueckgang": _skala(-df["rueckgang"], 0, b["rueckgang"][1]),
        "linie200": _skala(-df["abstand200"], 0, b["linie200"][1]),
        "rsi": _skala(-df["rsi_w"], -50, -b["rsi"][1]),
        "trend": _skala(-df["trend_z"], 0, b["trend"][1]),
        "stabil": pd.Series(np.where((close > df["sma20"]) & (df["rueckgang"] <= -0.05), 1.0, 0.0), index=df.index),
    }
    for k, v in teile.items():
        df[f"p_{k}"] = v * b[k][0]
    df["kurs_score"] = sum(df[f"p_{k}"] for k in teile)
    df.loc[df["sma200"].isna() | df["trend_z"].isna(), "kurs_score"] = np.nan

    df["golden_cross"] = (df["sma50"] > df["sma200"]) & (df["sma50"].shift() <= df["sma200"].shift())
    df["death_cross"] = (df["sma50"] < df["sma200"]) & (df["sma50"].shift() >= df["sma200"].shift())
    return df


def bewertungs_punkte(firma: dict, kurs_usd: float) -> tuple[float, list[str]]:
    """0–20 Punkte aus Fundamentaldaten + kurze Begründungen."""
    punkte, gruende = 0.0, []
    kgv_e = firma.get("kgv_erwartet")
    if kgv_e and kgv_e > 0:
        p = float(_skala(-kgv_e, -45, -15)) * 8          # KGV 45+ -> 0, 15 -> 8
        punkte += p
        gruende.append(f"erwartetes KGV {kgv_e:.0f}")
    ziel = firma.get("kursziel")
    if ziel and kurs_usd:
        potenzial = ziel / kurs_usd - 1
        punkte += float(_skala(potenzial, 0, 0.30)) * 6    # 0 % -> 0, 30 % -> 6
        gruende.append(f"Analysten-Kursziel {potenzial * 100:+.0f} %")
    wachstum = firma.get("gewinnwachstum")
    if wachstum is not None:
        punkte += float(_skala(wachstum, 0, 0.25)) * 6     # kein Wachstum -> 0, 25 % -> 6
        gruende.append(f"Gewinnwachstum {wachstum * 100:+.0f} %")
    return punkte, gruende


def gesamt_score_aktie(kurs_score: float, bewertung: float) -> float:
    """Kurs-Score (0–100) zählt 80 %, Bewertung (0–20) wird addiert."""
    return kurs_score * 0.8 + bewertung


def wertentwicklung(close: pd.Series) -> dict:
    letzter = close.iloc[-1]
    def ret(tage):
        return letzter / close.iloc[-tage - 1] - 1 if len(close) > tage else float("nan")
    jahr_start = close[close.index < pd.Timestamp(close.index[-1].year, 1, 1)]
    ytd = letzter / jahr_start.iloc[-1] - 1 if len(jahr_start) else float("nan")
    return {"1T": ret(1), "1W": ret(5), "1M": ret(21), "3M": ret(63), "lfd. Jahr": ytd, "1J": ret(252)}


# ----------------------------------------------------------------------------
# Rückblick-Test
# ----------------------------------------------------------------------------

def rueckblick(df: pd.DataFrame, schwelle: float) -> dict:
    """Hätte der Kurs-Score in der Vergangenheit geholfen?"""
    close, score = df["kurs"], df["kurs_score"]
    fwd = close.shift(-TRADING_DAYS) / close - 1
    ok = score.notna() & fwd.notna()
    sig = ok & (score >= schwelle)
    res = {
        "tage": int(ok.sum()), "signaltage": int(sig.sum()),
        "rendite_alle": float(fwd[ok].mean()) if ok.any() else float("nan"),
        "rendite_signal": float(fwd[sig].mean()) if sig.any() else float("nan"),
        "positiv_alle": float((fwd[ok] > 0).mean()) if ok.any() else float("nan"),
        "positiv_signal": float((fwd[sig] > 0).mean()) if sig.any() else float("nan"),
    }
    # Gleiches Geld, zwei Wege: A) 100 €/Monat sofort, B) 70 € sofort + 30 € Reserve bis Signal
    s = df[score.notna()]
    if len(s) < TRADING_DAYS:
        return res
    monatsanfang = set(s.groupby([s.index.year, s.index.month]).head(1).index)
    a = b = reserve = eingezahlt = 0.0
    for tag, kurs, sc in zip(s.index, s["kurs"], s["kurs_score"]):
        if tag in monatsanfang:
            eingezahlt += 100
            a += 100 / kurs
            b += 70 / kurs
            reserve += 30
        if reserve and sc >= schwelle:
            b += reserve / kurs
            reserve = 0.0
    letzter = s["kurs"].iloc[-1]
    res.update({"eingezahlt": eingezahlt, "wert_sparplan": a * letzter,
                "wert_signal": b * letzter + reserve, "reserve": reserve})
    return res


# ----------------------------------------------------------------------------
# Marktklima (Ampel)
# ----------------------------------------------------------------------------

def _k(x):
    return f"{abs(x):.1f}".replace(".", ",")


def marktklima(vix: float | None, sp500: pd.Series | None, breite_3m: float | None) -> dict:
    """
    Fasst das Umfeld in einer Ampel zusammen.
      VIX: < 18 ruhig, 18–25 nervös, > 25 Angst
      S&P 500 über/unter 200-Tage-Linie: Trend intakt / gebrochen
      Marktbreite: Gleichgewichteter S&P 500 vs. normaler S&P 500 (3 Monate)
    """
    punkte, gruende = 0, []
    if vix is not None and not math.isnan(vix):
        if vix > 25:
            punkte += 2; gruende.append(f"VIX {vix:.0f}: hohe Angst an den Märkten")
        elif vix > 18:
            punkte += 1; gruende.append(f"VIX {vix:.0f}: Anleger sind nervös")
        else:
            gruende.append(f"VIX {vix:.0f}: ruhige Lage")
    if sp500 is not None and len(sp500) > 200:
        sma = sp500.rolling(200).mean().iloc[-1]
        abstand = sp500.iloc[-1] / sma - 1
        if abstand < 0:
            punkte += 2; gruende.append(f"S&P 500 {_k(abstand * 100)} % unter seiner 200-Tage-Linie: Aufwärtstrend gebrochen")
        else:
            gruende.append(f"S&P 500 {_k(abstand * 100)} % über seiner 200-Tage-Linie: Aufwärtstrend intakt")
    if breite_3m is not None and not math.isnan(breite_3m):
        if breite_3m < -0.03:
            punkte += 1; gruende.append("Marktbreite schwach: wenige Großkonzerne tragen den Markt")
        elif breite_3m > 0.03:
            gruende.append("Marktbreite gut: der breite Markt zieht mit")
        else:
            gruende.append("Marktbreite neutral")

    if punkte >= 3:
        farbe, titel = "#cf222e", "Rot – Angst / Abwärtstrend"
        rat = ("Phasen wie diese waren langfristig oft gute Einstiegszeiten, können aber noch tiefer gehen. "
               "Sparplan weiterlaufen lassen, Extra-Käufe nur in mehreren kleinen Schritten.")
    elif punkte >= 1:
        farbe, titel = "#d4a72c", "Gelb – Nervöser Markt"
        rat = ("Erhöhte Unsicherheit. Sparplan normal weiter; Extra-Käufe nur bei hohem Score und gestaffelt.")
    else:
        farbe, titel = "#1a7f37", "Grün – Ruhiger Markt, Trend intakt"
        rat = "Ruhiges Umfeld. Sparplan normal weiter; Kurse sind meist nicht besonders günstig."
    return {"farbe": farbe, "titel": titel, "gruende": gruende, "rat": rat}
