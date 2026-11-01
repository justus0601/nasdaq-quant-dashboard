"""Nasdaq-100 ETF backtest and Monte Carlo learning dashboard."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nasdaq Quant Dashboard", page_icon="📈", layout="wide")
st.title("📈 Nasdaq Quant Dashboard")
st.caption("Nasdaq-100-ETF (SXRV.DE, EUR) · Historischer Backtest & Monte-Carlo-Risikoanalyse")
st.info("Lernprojekt, keine Anlageberatung. Historische Kurse sind keine Garantie für zukünftige Ergebnisse. Alle Ergebnisse vor Steuern.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(start_iso: str, end_iso: str) -> pd.Series:
    data = yf.download("SXRV.DE", start=start_iso, end=end_iso, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError("Keine Daten von Yahoo Finance erhalten. Bitte später erneut versuchen.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close.name = "Preis"
    if len(close) < 210:
        raise ValueError("Zu wenige Kursdaten für den SMA-200-Backtest.")
    return close


def calc_backtest(prices: pd.Series, start: date, cost: float, fast: int, slow: int) -> pd.DataFrame:
    df = pd.DataFrame({"Preis": prices})
    df["SMA_kurz"] = df["Preis"].rolling(fast).mean()
    df["SMA_lang"] = df["Preis"].rolling(slow).mean()
    df["Signal"] = (df["SMA_kurz"] > df["SMA_lang"]).astype(float)
    df.loc[df["SMA_lang"].isna(), "Signal"] = np.nan
    # Conservative close-based execution convention: signal at close t,
    # trade at close t+1, earn position return from t+1 to t+2.
    df["Position"] = df["Signal"].shift(2)
    df["Rendite"] = df["Preis"].pct_change()
    df = df.loc[df.index.date >= start].dropna(subset=["Position", "Rendite"]).copy()
    if len(df) < 2:
        raise ValueError("Nicht genug Backtest-Tage nach dem gewählten Startdatum.")
    df.iloc[0, df.columns.get_loc("Rendite")] = 0.0
    df["Wechsel"] = df["Position"].diff().abs()
    df.iloc[0, df.columns.get_loc("Wechsel")] = df["Position"].iloc[0]
    # Start at closing price of first included trading day.
    df["BH_Faktor"] = (1 + df["Rendite"]).cumprod() * (1 - cost)
    df["MA_Tag"] = (1 + df["Position"] * df["Rendite"]) * (1 - cost * df["Wechsel"]) - 1
    df["MA_Faktor"] = (1 + df["MA_Tag"]).cumprod()
    return df


def metrics(factor: pd.Series, initial: float) -> dict:
    years = max((factor.index[-1] - factor.index[0]).days / 365.25, 1 / 365.25)
    end = initial * factor.iloc[-1]
    peaks = factor.cummax().clip(lower=1.0)
    dd = factor / peaks - 1
    daily = factor.pct_change().dropna()
    # 2% fixed annual risk-free assumption, as in tutorial
    sharpe = ((daily.mean() * 252 - .02) / (daily.std() * np.sqrt(252))) if len(daily) > 2 and daily.std() > 0 else np.nan
    return {"Endwert": end, "Gewinn": end - initial, "Gesamtrendite": factor.iloc[-1] - 1,
            "CAGR": factor.iloc[-1] ** (1 / years) - 1, "Max. Drawdown": dd.min(), "Sharpe (2% rf)": sharpe}


def euro(x):
    return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

with st.sidebar:
    st.header("⚙️ Einstellungen")
    years = st.slider("Backtest-Zeitraum (Jahre)", 3, 20, 10)
    amount = st.number_input("Investitionsbetrag (€)", min_value=1000, max_value=10000000, value=10000, step=1000)
    compare_30k = st.checkbox("Auch 30.000 € vergleichen", value=True)
    fee_pct = st.slider("Kosten je Transaktion (%)", 0.0, 1.0, 0.1, 0.05)
    fast = st.slider("Kurzer SMA (Tage)", 10, 100, 50, 5)
    slow = st.slider("Langer SMA (Tage)", 120, 300, 200, 10)
    st.divider()
    simulations = st.select_slider("Monte-Carlo-Simulationen", options=[1000, 5000, 10000, 20000], value=10000)
    horizon = st.slider("Monte-Carlo-Horizont (Handelstage)", 5, 252, 21)
    seed = st.number_input("Zufalls-Seed", value=42, min_value=0, step=1)

if fast >= slow:
    st.error("Der kurze SMA muss kleiner als der lange SMA sein.")
    st.stop()

end = date.today()
start = date(end.year - years, end.month, min(end.day, 28))
# Extra warmup for long SMA, plus execution delay
warmup = start - timedelta(days=max(450, int(slow * 2.2)))
try:
    with st.spinner("Lade echte historische ETF-Kurse …"):
        prices = load_prices(warmup.isoformat(), (end + timedelta(days=1)).isoformat())
    bt = calc_backtest(prices, start, fee_pct / 100, fast, slow)
except Exception as exc:
    st.error(f"Daten-/Berechnungsfehler: {exc}")
    st.stop()

st.caption(f"Daten: {bt.index[0].date()} bis {bt.index[-1].date()} · {len(bt)} Handelstage · Xetra-Notierung in EUR · angepasste Schlusskurse")

tab1, tab2, tab3 = st.tabs(["📊 Backtest", "🎲 Monte Carlo", "ℹ️ Methodik"])

with tab1:
    amounts = [int(amount)]
    if compare_30k and int(amount) != 30000:
        amounts.append(30000)
    fig = go.Figure()
    for capital in amounts:
        for key, label, dash in [("BH_Faktor", "Buy & Hold", "solid"), ("MA_Faktor", f"SMA {fast}/{slow}", "dash")]:
            fig.add_trace(go.Scatter(x=bt.index, y=capital * bt[key], mode="lines", name=f"{euro(capital)} – {label}", line=dict(width=2.5, dash=dash), hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f} €<extra>%{fullData.name}</extra>"))
    fig.update_layout(title="Depotwertentwicklung", xaxis_title="Datum", yaxis_title="Depotwert (€)", hovermode="x unified", height=520)
    st.plotly_chart(fig, use_container_width=True)
    results = []
    for capital in amounts:
        for key, name in [("BH_Faktor", "Buy & Hold"), ("MA_Faktor", f"SMA {fast}/{slow}")]:
            results.append({"Startkapital (€)": capital, "Strategie": name, **metrics(bt[key], capital)})
    table = pd.DataFrame(results)
    st.subheader("Ergebnisse")
    st.dataframe(table.style.format({"Endwert": euro, "Gewinn": euro, "Gesamtrendite": "{:.2%}", "CAGR": "{:.2%}", "Max. Drawdown": "{:.2%}", "Sharpe (2% rf)": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.download_button("Ergebnisse als CSV herunterladen", table.to_csv(index=False).encode("utf-8-sig"), "backtest_ergebnisse.csv", "text/csv")
    dd_fig = go.Figure()
    for key, name in [("BH_Faktor", "Buy & Hold"), ("MA_Faktor", f"SMA {fast}/{slow}")]:
        factor = bt[key]
        drawdown = 100 * (factor / factor.cummax().clip(lower=1.0) - 1)
        dd_fig.add_trace(go.Scatter(x=bt.index, y=drawdown, name=name, mode="lines", line=dict(width=2)))
    dd_fig.update_layout(title="Drawdown", yaxis_title="Rückgang vom Höchststand (%)", height=380, hovermode="x unified")
    st.plotly_chart(dd_fig, use_container_width=True)
    st.download_button("Backtest-Zeitreihe als CSV", bt.to_csv().encode("utf-8-sig"), "backtest_zeitreihe.csv", "text/csv")

with tab2:
    st.write("Monte Carlo verwendet die historischen Tagesrenditen des gewählten Backtest-Zeitraums und simuliert unabhängige, normalverteilte Tagesrenditen. Das ist ein Modell, keine Prognose.")
    returns = bt["Rendite"].iloc[1:].dropna()
    mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
    rng = np.random.default_rng(int(seed))
    daily_sim = rng.normal(mu, sigma, size=(int(simulations), int(horizon)))
    # Avoid negative gross returns, practically irrelevant for current sigma
    daily_sim = np.maximum(daily_sim, -0.999)
    factors = np.column_stack([np.ones(int(simulations)), np.cumprod(1 + daily_sim, axis=1)])
    fig_mc = go.Figure()
    shown = min(100, int(simulations))
    for i in range(shown):
        fig_mc.add_trace(go.Scatter(x=np.arange(horizon + 1), y=amount * factors[i], mode="lines", line=dict(color="rgba(70,130,180,0.28)", width=1.1), showlegend=False, hovertemplate="Tag %{x}<br>%{y:,.2f} €<extra>Simulierter Pfad</extra>"))
    for label, values, color in [("Durchschnitt", factors.mean(axis=0), "#dc2626"), ("5%-Quantil", np.percentile(factors, 5, axis=0), "#ea580c"), ("95%-Quantil", np.percentile(factors, 95, axis=0), "#15803d")]:
        fig_mc.add_trace(go.Scatter(x=np.arange(horizon + 1), y=amount * values, mode="lines", name=label, line=dict(color=color, width=3)))
    fig_mc.update_layout(title=f"{simulations:,} Szenarien – {euro(amount)} (100 Pfade dargestellt)".replace(",", "."), xaxis_title="Handelstage", yaxis_title="Depotwert (€)", height=550)
    st.plotly_chart(fig_mc, use_container_width=True)
    st.caption(f"Geschätzte Tagesrendite: {mu:.4%} · Tagesvolatilität: {sigma:.4%} · 95%-Konfidenzniveau")
    # P&L loss convention: negative losses mean profits; VaR is non-negative floor
    loss_fraction = 1 - factors[:, -1]
    cutoff = float(np.quantile(loss_fraction, .95))
    tail = loss_fraction[loss_fraction >= cutoff]
    es_fraction = float(tail.mean())
    rows = []
    for capital in amounts:
        rows.append({"Startkapital (€)": capital, "Mittlerer Endwert (€)": capital * factors[:, -1].mean(), "5%-Endwert (€)": capital * np.percentile(factors[:, -1], 5), "VaR 95% (€)": capital * max(0, cutoff), "ES 95% (€)": capital * es_fraction, "Verlustwahrscheinlichkeit": float(np.mean(factors[:, -1] < 1))})
    risk_table = pd.DataFrame(rows)
    st.subheader("Risiko nach dem gewählten Zeitraum")
    st.dataframe(risk_table.style.format({"Mittlerer Endwert (€)": euro, "5%-Endwert (€)": euro, "VaR 95% (€)": euro, "ES 95% (€)": euro, "Verlustwahrscheinlichkeit": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.download_button("Monte-Carlo-Ergebnisse als CSV", risk_table.to_csv(index=False).encode("utf-8-sig"), "monte_carlo_risiko.csv", "text/csv")
    hist = go.Figure(go.Histogram(x=amount * factors[:, -1], nbinsx=70, name="Simulierte Endwerte"))
    hist.add_vline(x=amount, line_dash="dash", line_color="red")
    hist.update_layout(title="Verteilung der simulierten Endwerte", xaxis_title="Depotwert (€)", yaxis_title="Häufigkeit", height=400)
    st.plotly_chart(hist, use_container_width=True)

with tab3:
    st.markdown("""
    **Quelle:** Yahoo Finance via `yfinance`, Ticker `SXRV.DE` (EUR-Notierung des thesaurierenden iShares NASDAQ 100 UCITS ETF). `auto_adjust=True`.

    **Backtest:** Der SMA wird aus historischen Schlusskursen berechnet. Das Signal wird um zwei Handelstage für die Tagesrendite verzögert: Signal am Schlusskurs von Tag *t*, modellierte Ausführung am Schlusskurs *t+1*, Renditewirkung ab *t+2*. Diese Vereinfachung verhindert die Nutzung unbekannter Tages-Schlusskurse. Reale Ausführungskurse können abweichen.

    **Gebühren:** Eine einstellbare Pauschale pro Positionswechsel (Gebühren und Slippage zusammen). Buy & Hold zahlt einmalig Einstiegskosten. Keine Steuern, Depotkosten, ETF-spezifischen zusätzlichen Gebühren oder Cash-Verzinsung. ETF-laufende Kosten sind grundsätzlich bereits im historischen Fondskurs reflektiert.

    **Monte Carlo:** Aus den täglichen historischen Renditen im Backtest-Fenster werden Mittelwert und Standardabweichung geschätzt. Es werden unabhängige normalverteilte Tagesrenditen gezogen und geometrisch verkettet. **VaR 95 %** ist die Verlustschwelle, die in etwa 5 % der simulierten Fälle überschritten wird. **ES 95 %** ist der durchschnittliche Verlust der schlechtesten 5 %. Positive VaR-Werte sind als Verlustbeträge ausgewiesen; wenn das 95%-Verlustquantil negativ ist, wird VaR auf null begrenzt.

    **Grenzen:** Normalverteilung kann Extremereignisse unterschätzen. Ergebnisse hängen stark von Stichprobe, Parametern und Datenqualität ab. Die App simuliert einen ETF, daher entfällt die Korrelations-/Cholesky-Analyse des Mehr-Aktien-Projekts aus dem Lernmaterial. Kein Walk-forward-Test, keine echte Orderausführung, keine Handelsberatung.
    """)
