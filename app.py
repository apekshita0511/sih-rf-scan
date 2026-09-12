"""Streamlit entrypoint.

    streamlit run app.py

All logic lives in rfscan.app.dashboard / rfscan.app.live_simulation /
rfscan.visualization.plots -- this file only starts the app, mirroring how
main.py is a thin wrapper around rfscan.cli.
"""

from rfscan.app.dashboard import main

main()
