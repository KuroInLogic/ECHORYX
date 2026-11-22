import streamlit as st
import pandas as pd
import html
from pathlib import Path
from services.popularity_service import recommend_for_user
from services.profile_service import recommend_from_preferences


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ECHORYX | Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# STYLING
# =========================================================

def load_css():
    css_path = PROJECT_ROOT / "assets" / "style.css"

    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


load_css()


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_tracks():
    return pd.read_parquet(DATA_DIR / "tracks.parquet")


@st.cache_data
def load_training_data():
    return pd.read_parquet(DATA_DIR / "train.parquet")


@st.cache_data
def load_popularity():
    return pd.read_parquet(DATA_DIR / "popularity_ranking.parquet")


tracks = load_tracks()
train = load_training_data()
popularity = load_popularity()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def section_heading(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.caption(subtitle)


def display_track_cards(data, limit=10):
    """Display tracks as visual recommendation cards."""

    if data.empty:
        st.info("No tracks found.")
        return

    data = data.head(limit).reset_index(drop=True)

    for start in range(0, len(data), 2):
        columns = st.columns(2)

        for offset, column in enumerate(columns):
            index = start + offset

            if index >= len(data):
                break

            track = data.iloc[index]

            artist = html.escape(str(track.get("artist_name", "Unknown Artist")))
            title = html.escape(str(track.get("track_name", "Unknown Track")))

            listeners = track.get("listener_count")
            plays = track.get("total_plays")
            score = track.get("preference_score")

            with column:
                st.markdown(
                    f"""
                    <div class="feature-card track-card">
                        <div class="track-art">
                            <span>♫</span>
                        </div>
                        <div class="track-details">
                            <div class="track-title">{title}</div>
                            <div class="track-artist">{artist}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                metadata = []

                if pd.notna(listeners):
                    metadata.append(f"{int(listeners):,} listeners")

                if pd.notna(plays):
                    metadata.append(f"{int(plays):,} plays")

                if pd.notna(score):
                    metadata.append(f"Score: {float(score):.1f}")

                if metadata:
                    st.caption(" · ".join(metadata))


def display_track_table(data, limit=20):
    """Display searchable music catalogue results."""

    if data.empty:
        st.info("No tracks found.")
        return

    display = data.head(limit).copy()

    columns = [
        column for column in
        ["artist_name", "track_name", "listener_count", "total_plays", "play_count"]
        if column in display.columns
    ]

    display = display[columns].rename(columns={
        "artist_name": "Artist",
        "track_name": "Track",
        "listener_count": "Listeners",
        "total_plays": "Total Plays",
        "play_count": "Your Plays"
    })

    st.dataframe(display, width="stretch", hide_index=True)


# =========================================================
# SIDEBAR
# =========================================================

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
            "Create Your Profile",
            "Mood Studio",
            "My Library",
            "Model Insights"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("POWERED BY LAST.FM 1K")
    st.caption("ECHORYX · 2026")


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">YOUR MUSIC UNIVERSE</div>
        <h1>Discover beyond your rhythm.</h1>
        <p>
            Explore familiar favourites, find unexpected sounds
            and let your music journey take a new direction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    section_heading(
        "The universe behind ECHORYX",
        "A glimpse into the Last.fm listening community."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Tracks", f"{len(tracks):,}")
    c2.metric("Listeners", f"{train['user_idx'].nunique():,}")
    c3.metric("Interactions", f"{len(train):,}")

    section_heading(
        "Trending across the universe",
        "Tracks with the highest listener support in our dataset."
    )

    display_track_cards(popularity, 6)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>♫ Discover music</h3>
            <p>
                Explore recommendations based on historical
                listening patterns from the Last.fm community.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✧ Build your sound</h3>
            <p>
                Select your favourite artists and create a
                music discovery profile of your own.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# EXPLORE
# =========================================================

elif page == "Explore":

    st.markdown("# Explore")
    st.caption("Find artists and tracks from the Last.fm collection.")

    search = st.text_input(
        "Search music",
        placeholder="Search by track or artist..."
    )

    if search:

        results = tracks[
            tracks["track_name"].str.contains(
                search, case=False, na=False
            )
            |
            tracks["artist_name"].str.contains(
                search, case=False, na=False
            )
        ]

        st.write(f"Found {len(results):,} matching tracks.")
        display_track_table(results, 50)

    else:

        section_heading("Popular tracks")
        display_track_cards(popularity, 10)


# =========================================================
# DISCOVER — EXISTING LAST.FM USERS
# =========================================================

elif page == "Discover":

    st.markdown("# Discover")
    st.caption("Explore music through existing Last.fm listening profiles.")

    user_ids = sorted(train["user_idx"].unique().tolist())

    selected_user = st.selectbox(
        "Select a listener profile",
        user_ids,
        format_func=lambda x: f"Listener {x}"
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    if st.button("Generate recommendations", type="primary"):

        recommendations = recommend_for_user(
            int(selected_user),
            n=recommendation_count
        )

        st.session_state["demo_recommendations"] = recommendations
        st.session_state["demo_recommendation_user"] = selected_user

    if "demo_recommendations" in st.session_state:

        if st.session_state["demo_recommendation_user"] == selected_user:

            recommendations = st.session_state["demo_recommendations"]

            section_heading(
                "Recommended for this listener",
                "Based on global popularity, excluding tracks in the listener's training history."
            )

            display_track_cards(recommendations, recommendation_count)


# =========================================================
# CREATE YOUR PROFILE — NEW USERS
# =========================================================

elif page == "Create Your Profile":

    st.markdown("# Create Your Music Profile")

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">YOUR SOUND. YOUR DISCOVERY.</div>
        <h1>What do you love listening to?</h1>
        <p>
            Pick your favourite artists. ECHORYX will explore
            the listening habits of similar music fans to
            discover tracks you might enjoy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Get available artist names.
    available_artists = sorted(
        tracks["artist_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist(),
        key=str.casefold
    )

    if "profile_artists" not in st.session_state:
        st.session_state["profile_artists"] = []

    if "profile_recommendations" not in st.session_state:
        st.session_state["profile_recommendations"] = None

    section_heading(
        "Build your taste profile",
        "Start with at least one artist. Three to five artists will give the system more preference signals."
    )

    artist_search = st.text_input(
        "Find an artist",
        placeholder="Search for Radiohead, Coldplay, Muse..."
    )

    if artist_search:

        matching_artists = [
            artist for artist in available_artists
            if artist_search.casefold() in artist.casefold()
            and artist not in st.session_state["profile_artists"]
        ][:50]

        if matching_artists:

            selected_artist = st.selectbox(
                "Matching artists",
                matching_artists,
                key="artist_search_selection"
            )

            if st.button("Add artist", key="add_profile_artist"):

                if selected_artist not in st.session_state["profile_artists"]:
                    st.session_state["profile_artists"].append(selected_artist)
                    st.session_state["profile_recommendations"] = None

                st.rerun()

        else:
            st.info("No matching artists found.")

    section_heading("Your selected artists")

    selected_artists = st.session_state["profile_artists"]

    if selected_artists:

        for artist in selected_artists:

            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="feature-card"
                         style="min-height:0; padding:12px 18px; margin-bottom:8px;">
                        <strong>{html.escape(artist)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if st.button(
                    "Remove",
                    key=f"remove_{artist}"
                ):
                    st.session_state["profile_artists"].remove(artist)
                    st.session_state["profile_recommendations"] = None
                    st.rerun()

    else:
        st.info("You haven't selected any artists yet.")

    st.write("")

    recommendation_count = st.slider(
        "How many tracks would you like?",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="profile_recommendation_count"
    )

    if st.button(
        "Build My Music Profile",
        type="primary",
        disabled=len(selected_artists) == 0
    ):

        with st.spinner("Exploring listening patterns..."):

            recommendations, statistics = recommend_from_preferences(
                selected_artists=selected_artists,
                train=train,
                tracks=tracks,
                n=recommendation_count
            )

        st.session_state["profile_recommendations"] = recommendations
        st.session_state["profile_statistics"] = statistics
        st.session_state["profile_generated_for"] = list(selected_artists)

    # Display generated profile.
    recommendations = st.session_state["profile_recommendations"]

    if recommendations is not None:

        st.divider()

        section_heading(
            "Your discovery results",
            "Recommendations generated from listeners sharing your selected artist preferences."
        )

        statistics = st.session_state["profile_statistics"]

        c1, c2 = st.columns(2)

        c1.metric(
            "Similar listeners",
            f"{statistics['similar_listeners']:,}"
        )

        c2.metric(
            "Matched artists",
            f"{statistics['matched_artists']:,}"
        )

        if not recommendations.empty:

            display_track_cards(
                recommendations,
                recommendation_count
            )

            st.caption(
                "These recommendations are inferred from historical "
                "listening patterns. They are not based on a live "
                "Spotify or Last.fm account."
            )

        else:
            st.warning(
                "We couldn't find enough matching listening patterns. "
                "Try selecting different artists."
            )


# =========================================================
# MOOD STUDIO
# =========================================================

elif page == "Mood Studio":

    st.markdown("# Mood Studio")

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">COMING SOON</div>
        <h1>Music for every state of mind.</h1>
        <p>
            Explore a different way of discovering music
            through moods and listening experiences.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Mood-based recommendations are planned for a future version."
    )


# =========================================================
# MY LIBRARY
# =========================================================

elif page == "My Library":

    st.markdown("# My Library")
    st.caption("Explore the historical listening activity of a Last.fm user.")

    user_ids = sorted(train["user_idx"].unique().tolist())

    selected_user = st.selectbox(
        "Select listener",
        user_ids,
        format_func=lambda x: f"Listener {x}",
        key="library_user"
    )

    user_history = train[
        train["user_idx"] == selected_user
    ]

    library = user_history.merge(
        tracks,
        on="track_idx",
        how="left"
    )

    library = library.sort_values(
        "play_count",
        ascending=False
    )

    c1, c2 = st.columns(2)

    c1.metric("Unique tracks", f"{len(library):,}")
    c2.metric("Total plays", f"{library['play_count'].sum():,}")

    section_heading("Listening history")

    display_track_table(library, 50)


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.markdown("# Model Insights")
    st.caption("Understanding how the ECHORYX recommendation engine performs.")

    section_heading(
        "Popularity-based recommendation",
        "Current evaluation results from the held-out temporal test set."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Precision@10", "0.1302")
    c2.metric("Recall@10", "0.0025")
    c3.metric("NDCG@10", "0.1348")

    st.divider()

    section_heading("Evaluation statistics")

    evaluation_data = pd.DataFrame({
        "Metric": [
            "Users evaluated",
            "Recommendation hits",
            "Test interactions",
            "Unique tracks recommended",
            "Unique cold-start tracks",
            "Cold-start test interactions"
        ],
        "Value": [
            "951",
            "1,238",
            "552,608",
            "49",
            "304",
            "1,860"
        ]
    })

    st.dataframe(
        evaluation_data,
        width="stretch",
        hide_index=True
    )

    section_heading("How does it work?")

    st.markdown("""
    The current baseline ranks tracks using:

    - Unique listener count.
    - Total listening plays.

    Tracks already present in the selected listener's training
    history are excluded.

    The popularity statistics are calculated exclusively
    from the training dataset.
    """)

    st.info(
        "The Create Your Music Profile feature uses a separate "
        "preference-based recommendation approach."
    )
