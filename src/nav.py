# src/nav.py
import streamlit as st

def sidebar_nav(active: str = "Home"):
    st.sidebar.title("Macro Engine")
    st.sidebar.caption("Navigation")

    st.sidebar.page_link("app.py", label="Home", icon="🏠", disabled=(active == "Home"))
    st.sidebar.page_link("pages/2_Macro_Charts.py", label="Macro charts", icon="📊", disabled=(active == "Macro charts"))
    st.sidebar.page_link("pages/1_Regime_Deep_Dive.py", label="Regime deep dive", icon="🧭", disabled=(active == "Regime deep dive"))
    st.sidebar.page_link("pages/5_Drivers.py", label="Drivers", icon="🧩", disabled=(active == "Drivers"))
    st.sidebar.page_link("pages/3_Ticker_Detail.py", label="Ticker drilldown", icon="🔎", disabled=(active == "Ticker drilldown"))