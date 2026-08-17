# Full NSE VCP Scanner

Android/browser-oriented Streamlit app.

Features:
- Attempts to load the current NSE equity symbol list from NSE archives.
- Scans the selected NSE universe using daily history.
- Minervini-inspired trend template.
- Three contraction measurements.
- ATR contraction.
- Volume dry-up.
- Pivot proximity.
- Breakout + volume.
- VCP score and CSV export.

Run:
pip install -r requirements.txt
streamlit run app.py

For browser use, deploy the repository to Streamlit Community Cloud.

Important:
- This is a rule-based screening prototype, not financial advice.
- Yahoo Finance data may be delayed/incomplete.
- NSE/Yahoo endpoints can change or rate-limit requests.
- Full-NSE scanning can be slow. Start with 100–500 stocks for testing.
- Before live use, replace the data layer with a reliable broker/data provider and backtest the exact rules.
