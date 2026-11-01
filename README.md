# Nasdaq Quant Dashboard

Interaktive Streamlit-App mit historischen Nasdaq-100-ETF-Kursen (`SXRV.DE`), Buy-and-Hold- und SMA-Crossover-Backtest sowie Monte-Carlo-Simulation mit VaR und Expected Shortfall.

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Online bereitstellen

1. Ein öffentliches GitHub-Repository anlegen.
2. `app.py` und `requirements.txt` in dessen Hauptverzeichnis hochladen.
3. Auf https://share.streamlit.io/ mit GitHub anmelden.
4. Neue App erstellen: Repository, Branch `main`, Entry point `app.py` auswählen.
5. Deploy klicken. Die App erhält einen öffentlichen Link.

Daten werden bei App-Start via Yahoo Finance bezogen und für eine Stunde zwischengespeichert. Für die App ist kein API-Key nötig; Datenabruf kann vorübergehend fehlschlagen.

**Hinweis:** Lernprojekt, keine Anlageberatung.
