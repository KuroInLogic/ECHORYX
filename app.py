import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ECHORYX | Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external stylesheet
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"

    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# Sidebar
with st.sidebar:
    st.markdown("# 🎧 ECHORYX")
    st.caption("MUSIC, UNDERSTOOD DIFFERENTLY.")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Explore",
            "Discover",
            "Mood Studio",
            "My Library",
            "Model Insights"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("ECHORYX · 2026")


# Homepage
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">YOUR MUSIC UNIVERSE</div>
        <h1>Discover beyond your rhythm.</h1>
        <p>
            Your next favourite track might be something
            you've never heard before.
        </p>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "Search music",
        placeholder="Search songs, artists, albums or genres..."
    )

    st.markdown(
        '<div class="section-title">Explore your sound</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>♫ Similar Sounds</h3>
            <p>Find tracks that share the characteristics of your favourites.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✧ Beyond Your Genre</h3>
            <p>Discover unfamiliar artists, genres and musical experiences.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Your discovery journey</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Your personalized recommendations will appear here "
        "once the recommendation engine is connected."
    )

else:
    st.markdown(f"# {page}")
    st.info(f"The {page} section is under development.")
