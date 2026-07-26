"""The web UI: start searches, watch them run, and compare the workflows found.

`server` serves the API and the page; `runner` is the subprocess that executes
one search. State lives on disk (see `flowopt.runstore`), so the
server holds nothing that a restart would lose.
"""
