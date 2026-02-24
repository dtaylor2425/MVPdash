import streamlit as st

st.set_page_config(
    page_title="Drivers",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Clean minimal styling
# -----------------------------

def inject_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }

        .block-container {
            max-width: 1000px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .page-title {
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 2px;
        }

        .page-sub {
            font-size: 13px;
            color: rgba(0,0,0,0.55);
            margin-bottom: 1.5rem;
        }

        .driver-title {
            font-size: 14px;
            font-weight: 800;
        }

        .driver-desc {
            font-size: 13px;
            color: rgba(0,0,0,0.65);
        }

        .stButton > button {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# -----------------------------
# Top navigation
# -----------------------------

nav = st.columns([1, 1, 6])

with nav[0]:
    if st.button("Home", use_container_width=True):
        st.switch_page("app.py")

with nav[1]:
    if st.button("Macro charts", use_container_width=True):
        st.switch_page("pages/2_Macro_Charts.py")

# -----------------------------
# Header
# -----------------------------

st.markdown("<div class='page-title'>Key Drivers</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-sub'>Primary forces shaping the current macro regime.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Drivers
# -----------------------------

drivers = [
    ("Equities favored by regime score",
     "Tight credit spreads and improving breadth support equity risk exposure.",
     "Credit spreads"),

    ("Breadth improving",
     "Small caps and equal weight outperforming signals expanding participation.",
     "Risk appetite"),

    ("Credit spreads tight",
     "High yield OAS contracting supports pro-risk positioning.",
     "Credit spreads"),

    ("Dollar weak",
     "Broad dollar softness supports commodities and non-US assets.",
     "Broad dollar"),

    ("Real yields elevated",
     "Higher real rates constrain duration and long-duration equity multiples.",
     "Real yields"),
]

col1, col2 = st.columns(2, gap="large")

for i, (title, desc, target) in enumerate(drivers):
    col = col1 if i % 2 == 0 else col2

    with col:
        with st.container(border=True):
            st.markdown(f"<div class='driver-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='driver-desc'>{desc}</div>", unsafe_allow_html=True)
            st.markdown("")

            if st.button("Open related chart",
                         key=f"driver_{i}",
                         use_container_width=True):
                st.session_state["selected_metric"] = target
                st.switch_page("pages/2_Macro_Charts.py")

# -----------------------------
# Regime Summary
# -----------------------------

st.markdown("")

with st.container(border=True):
    st.markdown("### Regime Summary")
    st.write(
        """
Current positioning reflects improving risk appetite, tight credit conditions,
and supportive breadth. Real yields remain a constraint on duration.
"""
    )