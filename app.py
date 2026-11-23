import streamlit as st
import pandas as pd
import html
from pathlib import Path
from services.explanation_service import explain_hybrid_recommendation
from services.user_cf_service import recommend_for_user
from services.profile_service import recommend_from_preferences
from services.ncf_service import recommend_for_user as recommend_with_ncf
from services.hybrid_service import recommend_hybrid


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ECHORYX | Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# STYLING
# =========================================================

def load_css():
    css_path = PROJECT_ROOT / "assets" / "style.css"

    if css_path.exists():
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

            artist = html.escape(
                str(track.get("artist_name", "Unknown Artist"))
            )

            title = html.escape(
                str(track.get("track_name", "Unknown Track"))
            )

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
                    metadata.append(
                        f"{int(listeners):,} listeners"
                    )

                if pd.notna(plays):
                    metadata.append(
                        f"{int(plays):,} plays"
                    )

                if pd.notna(score):
                    metadata.append(
                        f"Model score: {float(score):.4f}"
                    )

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
        [
            "artist_name",
            "track_name",
            "listener_count",
            "total_plays",
            "play_count"
        ]
        if column in display.columns
    ]

    display = display[columns].rename(columns={
        "artist_name": "Artist",
        "track_name": "Track",
        "listener_count": "Listeners",
        "total_plays": "Total Plays",
        "play_count": "Your Plays"
    })

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )


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
            "Neural Recommendations",
            "Hybrid Recommendations",
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
                search,
                case=False,
                na=False
            )
            |
            tracks["artist_name"].str.contains(
                search,
                case=False,
                na=False
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

    st.caption(
        "Explore personalised recommendations based on "
        "similar Last.fm listeners."
    )

    user_ids = sorted(train["user_idx"].unique().tolist())

    selected_user = st.selectbox(
        "Select a listener profile",
        user_ids,
        format_func=lambda x: f"Listener {x}",
        key="cf_user"
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="cf_recommendation_count"
    )


    st.markdown("### Personalise your discovery")

    available_artists = sorted(
        tracks["artist_name"].dropna().unique().tolist()
    )

    selected_artists = st.multiselect(
        "Which artists would you like to explore?",
        options=available_artists,
        max_selections=10,
        key="hybrid_selected_artists",
        placeholder="Search and select your favourite artists..."
    )

    discovery_intensity = st.slider(
        "Discovery preference influence",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help=(
            "Controls how strongly your selected artists "
            "influence the final recommendation ranking."
        ),
        key="hybrid_discovery_intensity"
    )


    if st.button(
        "Generate recommendations",
        type="primary",
        key="generate_cf"
    ):
        with st.spinner(
            "Finding listeners with similar music taste..."
        ):
            recommendations = recommend_for_user(
                user_idx=int(selected_user),
                train=train,
                tracks=tracks,
                n=recommendation_count
            )

        st.session_state["demo_recommendations"] = recommendations
        st.session_state["demo_recommendation_user"] = selected_user
        st.session_state["demo_recommendation_count"] = (
            recommendation_count
        )

    if (
        "demo_recommendations" in st.session_state
        and st.session_state.get("demo_recommendation_user")
        == selected_user
    ):
        recommendations = st.session_state["demo_recommendations"]

        section_heading(
            "Recommended for this listener",
            "Ranked using listening-pattern similarity and "
            "tracks enjoyed by similar listeners."
        )

        if not recommendations.empty:
            if recommendations["fallback_used"].any():
                st.info(
                    "ECHORYX is using popularity-based recommendations "
                    "because this listener has limited listening history. "
                    "Recommendations will become more personalised "
                    "as more listening data becomes available."
                )
            else:
                st.success(
                    "Personalised recommendations generated using "
                    "User-CF, NeuMF and popularity signals."
                )
            display_track_cards(
                recommendations,
                recommendation_count
            )

            st.caption(
                "Recommendations are generated using "
                "user-based collaborative filtering with "
                "cosine similarity."
            )

        else:
            st.warning(
                "We couldn't find suitable recommendations "
                "for this listener."
            )

            st.info(
                "Try selecting another listener profile."
            )

# =========================================================
# NEURAL RECOMMENDATIONS — NEUMF
# =========================================================

elif page == "Neural Recommendations":

    st.markdown("# Neural Recommendations")

    st.caption(
        "Discover tracks ranked by ECHORYX's Neural "
        "Collaborative Filtering model."
    )

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">POWERED BY NEUMF</div>
        <h1>Discover your next favourite.</h1>
        <p>
            NeuMF combines Generalised Matrix Factorisation
            and a Multi-Layer Perceptron to learn user-track
            interaction patterns from historical listening data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    user_ids = sorted(train["user_idx"].unique().tolist())

    selected_user = st.selectbox(
        "Select a listener profile",
        user_ids,
        format_func=lambda x: f"Listener {x}",
        key="neumf_user"
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="neumf_recommendation_count"
    )

    if st.button(
        "Generate neural recommendations",
        type="primary",
        key="generate_neumf"
    ):
        try:
            with st.spinner(
                "NeuMF is analysing listening patterns..."
            ):
                recommendations = recommend_with_ncf(
                    user_idx=int(selected_user),
                    train=train,
                    tracks=tracks,
                    n=recommendation_count
                )

            st.session_state["neumf_recommendations"] = (
                recommendations
            )

            st.session_state["neumf_recommendation_user"] = (
                selected_user
            )

            st.session_state["neumf_generated_count"] = (
                recommendation_count
            )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"Could not generate recommendations: {error}"
            )

    if (
        "neumf_recommendations" in st.session_state
        and st.session_state.get("neumf_recommendation_user")
        == selected_user
    ):
        recommendations = st.session_state[
            "neumf_recommendations"
        ]

        st.divider()

        section_heading(
            "Your neural recommendations",
            "Tracks ranked by the NeuMF model's predicted "
            "user-track interaction scores."
        )

        if not recommendations.empty:
            display_track_cards(
                recommendations,
                st.session_state.get(
                    "neumf_generated_count",
                    recommendation_count
                )
            )

            st.caption(
                "Previously heard training tracks are excluded. "
                "Scores represent model ranking outputs, not "
                "probabilities of user satisfaction."
            )

        else:
            st.info(
                "No unseen recommendations were available "
                "for this listener."
            )


# =========================================================
# HYBRID RECOMMENDATIONS — CF + NEUMF + POPULARITY
# =========================================================

elif page == "Hybrid Recommendations":

    st.markdown("# Hybrid Recommendations")
    st.caption(
        "A combined recommendation engine using listener similarity, "
        "neural preference scores and overall track popularity."
    )

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">THE ECHORYX HYBRID ENGINE</div>
        <h1>Three signals. One discovery experience.</h1>
        <p>
            Hybrid recommendations combine User-Based Collaborative Filtering,
            Neural Collaborative Filtering and popularity ranking into one list.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    user_ids = sorted(train["user_idx"].unique().tolist())

    selected_user = st.selectbox(
        "Select a listener profile",
        user_ids,
        format_func=lambda x: f"Listener {x}",
        key="hybrid_user"
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="hybrid_recommendation_count"
    )

    selected_artists = st.multiselect(
        "Choose your favourite artists",
        options=sorted(
            tracks["artist_name"].dropna().unique().tolist()
        ),
        default=[],
        help="Select artists to personalise your recommendations.",
        key="hybrid_selected_artists"
    )

    discovery_intensity = st.slider(
        "Artist preference influence",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Higher values give your selected artist preferences more influence.",
        key="hybrid_discovery_intensity"
    )

    if st.button(
        "Generate hybrid recommendations",
        type="primary",
        key="generate_hybrid"
    ):
        try:
            with st.spinner(
                "Combining collaborative, neural and popularity signals..."
            ):
                recommendations = recommend_hybrid(
                    user_idx=int(selected_user),
                    train=train,
                    max_per_artist=2,
                    tracks=tracks,
                    n=recommendation_count,
                    weights={
                        "user_cf": 0.60,
                        "ncf": 0.25,
                        "popularity": 0.15
                    },
                    selected_artists=selected_artists,
                    discovery_intensity=discovery_intensity
                )

            st.session_state["hybrid_recommendations"] = recommendations
            st.session_state["hybrid_recommendation_user"] = selected_user
            st.session_state["hybrid_generated_count"] = recommendation_count

        except Exception as error:
            st.error(f"Could not generate hybrid recommendations: {error}")

    if (
        "hybrid_recommendations" in st.session_state
        and st.session_state.get("hybrid_recommendation_user") == selected_user
    ):
        recommendations = st.session_state["hybrid_recommendations"]
        generated_count = st.session_state.get(
            "hybrid_generated_count", recommendation_count
        )

        section_heading(
            "Recommended for this listener",
            "Tracks ranked by the combined hybrid score."
        )

        if not recommendations.empty:
            display_track_cards(recommendations, generated_count)
            st.caption(
                "Hybrid weights: 60% User-Based CF · 25% NeuMF · "
                "15% Popularity. Tracks in the listener's training history "
                "are excluded."
            )
        else:
            st.info("No unseen recommendations were available for this listener.")


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
        "Start with at least one artist. Three to five artists "
        "will give the system more preference signals."
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

            if st.button(
                "Add artist",
                key="add_profile_artist"
            ):
                if selected_artist not in st.session_state["profile_artists"]:
                    st.session_state["profile_artists"].append(
                        selected_artist
                    )

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
                    st.session_state["profile_artists"].remove(
                        artist
                    )

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
            "Recommendations generated from listeners sharing "
            "your selected artist preferences."
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
            display_track_cards(recommendations, generated_count)

            st.divider()
            section_heading(
                "Why these tracks?",
                "Understand the signals behind each recommendation."
            )

            for index, (_, track) in enumerate(
                recommendations.iterrows(), start=1
            ):
                explanation = explain_hybrid_recommendation(track)

                track_name = track.get("track_name", f"Track {index}")
                artist_name = track.get("artist_name", "Unknown artist")

                with st.expander(
                    f"{index}. {track_name} — {artist_name}"
                ):
                    st.write(explanation["explanation"])

                    st.markdown("**Recommendation signal breakdown**")

                    scores = explanation["component_scores"]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "User-CF",
                        f'{scores["User-CF"]:.3f}'
                    )

                    col2.metric(
                        "NeuMF",
                        f'{scores["NeuMF"]:.3f}'
                    )

                    col3.metric(
                        "Popularity",
                        f'{scores["Popularity"]:.3f}'
                    )

                    st.metric(
                        "Final Hybrid Score",
                        f'{explanation["hybrid_score"]:.3f}'
                    )

                    st.caption(
                        f'Strongest contributing signal: '
                        f'{explanation["strongest_signal"]}'
                    )

            st.caption(
                "Hybrid weights: 60% User-Based CF · 25% NeuMF · "
                "15% Popularity. Tracks in the listener's training "
                "history are excluded."
            )

        else:
            st.info(
                "No unseen recommendations were available for this listener."
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

    st.caption(
        "Explore the historical listening activity of a Last.fm user."
    )

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

    c2.metric(
        "Total plays",
        f"{library['play_count'].sum():,}"
    )

    section_heading("Listening history")

    display_track_table(library, 50)

# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.markdown("# Model Insights")

    st.caption(
        "Explore the recommendation algorithms and their "
        "performance across the Last.fm dataset."
    )

    # -----------------------------------------------------
    # MODEL OVERVIEW
    # -----------------------------------------------------

    section_heading(
        "Recommendation models",
        "Four evaluated approaches powering music discovery."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>01 · Popularity-Based</h3>
            <p>
                Ranks tracks using unique listener count
                and total listening plays.
            </p>
            <p><strong>Status:</strong> Evaluated</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>02 · User-Based CF</h3>
            <p>
                Uses cosine similarity to identify listeners
                with similar music preferences.
            </p>
            <p><strong>Status:</strong> Evaluated</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>03 · Neural CF (NeuMF)</h3>
            <p>
                Combines matrix factorisation and neural
                network layers to learn user-track patterns.
            </p>
            <p><strong>Status:</strong> Evaluated</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <h3>04 · Hybrid</h3>
            <p>
                Combines User-Based CF, NeuMF and popularity
                ranking using weighted rank scores.
            </p>
            <p><strong>Status:</strong> Evaluated</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    
    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    section_heading(
        "Model performance comparison",
        "Popularity-Based vs User-Based CF vs NeuMF vs Hybrid."
    )

    st.caption(
        "Evaluation: 951 users · 175,242 candidate tracks · "
        "Top-10 recommendations per user."
    )

    st.caption(
        "Hybrid weights were selected through exploratory comparisons "
        "on this evaluation split. Treat these results as provisional "
        "until confirmed on a separate validation split."
    )

    comparison = pd.DataFrame({
        "Model": [
            "Popularity-Based",
            "User-Based CF",
            "NeuMF",
            "Hybrid (40/40/20)",
            "Hybrid (60/25/15)"
        ],
        "Precision@10": [
            0.1302, 0.1848, 0.1743428, 0.252366, 0.255415
        ],
        "Recall@10": [
            0.0025, 0.0038, 0.0036315, 0.005407, 0.005388
        ],
        "NDCG@10": [
            0.1348, 0.1866, 0.1769150, 0.264042, 0.267879
        ]
    })

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # METRIC VISUALISATION
    # -----------------------------------------------------

    section_heading(
        "Performance visualisation",
        "Compare how accurately each model ranks relevant tracks."
    )

    selected_metric = st.selectbox(
        "Select evaluation metric",
        ["Precision@10", "Recall@10", "NDCG@10"],
        key="insights_metric"
    )

    chart_data = comparison[["Model", selected_metric]].set_index("Model")

    st.bar_chart(chart_data)

    # -----------------------------------------------------
    # BEST MODEL BY METRIC
    # -----------------------------------------------------

    st.divider()

    section_heading(
        "Evaluation highlights",
        "The strongest model depends on the metric being measured."
    )

    best_precision = comparison.loc[
        comparison["Precision@10"].idxmax()
    ]

    best_recall = comparison.loc[
        comparison["Recall@10"].idxmax()
    ]

    best_ndcg = comparison.loc[
        comparison["NDCG@10"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Precision@10",
            best_precision["Model"],
            f'{best_precision["Precision@10"]:.4f}'
        )

    with col2:
        st.metric(
            "Best Recall@10",
            best_recall["Model"],
            f'{best_recall["Recall@10"]:.4f}'
        )

    with col3:
        st.metric(
            "Best NDCG@10",
            best_ndcg["Model"],
            f'{best_ndcg["NDCG@10"]:.4f}'
        )

    # -----------------------------------------------------
    # METRIC EXPLANATIONS
    # -----------------------------------------------------

    st.divider()

    section_heading(
        "Understanding the metrics",
        "What do these evaluation measurements actually mean?"
    )

    with st.expander("Precision@10"):
        st.write(
            "Measures the proportion of the top 10 recommended tracks "
            "that are relevant to the user's test listening history."
        )

    with st.expander("Recall@10"):
        st.write(
            "Measures the proportion of relevant test tracks that "
            "the model successfully retrieves in its top 10 recommendations."
        )

    with st.expander("NDCG@10"):
        st.write(
            "Measures ranking quality by rewarding relevant tracks "
            "that appear higher in the recommendation list."
        )


    # -----------------------------------------------------
    # PERFORMANCE HIGHLIGHTS
    # -----------------------------------------------------

    section_heading("Performance highlights")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best Precision@10",
        "0.2554",
        "Hybrid (60/25/15)"
    )

    c2.metric(
        "Best Recall@10",
        "0.0054",
        "Hybrid (40/40/20)"
    )

    c3.metric(
        "Best NDCG@10",
        "0.2679",
        "Hybrid (60/25/15)"
    )

    st.info(
        "The hybrid model with 60/25/15 weights achieves the "
        "strongest Precision@10 and NDCG@10 among the tested "
        "configurations. The 40/40/20 configuration has a slightly "
        "higher Recall@10. These are offline evaluation results."
    )

    # -----------------------------------------------------
    # PERFORMANCE VISUALIZATION
    # -----------------------------------------------------

    section_heading("Performance visualization")

    selected_metric = st.selectbox(
        "Select evaluation metric",
        [
            "Precision@10",
            "Recall@10",
            "NDCG@10"
        ],
        key="insights_metric_chart"
    )

    metric_values = {
        "Precision@10": [
            0.1302,
            0.1848,
            0.1743428,
            0.255415
        ],

        "Recall@10": [
            0.0025,
            0.0038,
            0.0036315,
            0.005388
        ],

        "NDCG@10": [
            0.1348,
            0.1866,
            0.1769150,
            0.267879
        ]
    }

    chart_data = pd.DataFrame({
        "Popularity-Based": [
            metric_values[selected_metric][0]
        ],

        "User-Based CF": [
            metric_values[selected_metric][1]
        ],

        "NeuMF": [
            metric_values[selected_metric][2]
        ],

        "Hybrid (60/25/15)": [
            metric_values[selected_metric][3]
        ]

    }, index=["Model Performance"])

    st.bar_chart(
        chart_data,
        horizontal=True
    )

    st.divider()

    # -----------------------------------------------------
    # RECOMMENDATION DIVERSITY
    # -----------------------------------------------------

    section_heading(
        "Recommendation diversity",
        "Number of distinct tracks recommended across all users."
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Popularity", "49")
    d2.metric("User-Based CF", "921")
    d3.metric("NeuMF", "2,887")
    d4.metric("Hybrid", "Not recorded")

    stats = pd.DataFrame({
        "Statistic": [
            "Users evaluated",
            "Recommendation hits",
            "Unique recommended tracks",
            "Test interactions",
            "Candidate catalogue size",
            "Recommendations per user"
        ],

        "Popularity-Based": [
            "951",
            "1,238",
            "49",
            "552,608",
            "175,242",
            "10"
        ],

        "User-Based CF": [
            "951",
            "1,757",
            "921",
            "552,608",
            "175,242",
            "10"
        ],

        "NeuMF": [
            "951",
            "1,658",
            "2,887",
            "552,608",
            "175,242",
            "10"
        ],

        "Hybrid (60/25/15)": [
            "951",
            "2,430",
            "Not recorded",
            "552,608",
            "175,242",
            "10"
        ]
    })

    st.dataframe(
        stats,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # ALGORITHM EXPLANATION
    # -----------------------------------------------------

    section_heading("Understanding the algorithms")

    with st.expander("01 · Popularity-Based Recommendation"):

        st.write(
            "The baseline model ranks tracks according to "
            "their unique listener count and total play count "
            "in the training dataset."
        )

        st.write(
            "Tracks already present in the target listener's "
            "training history are excluded."
        )

    with st.expander("02 · User-Based Collaborative Filtering"):

        st.write(
            "The model represents listeners using their "
            "binary track-listening histories."
        )

        st.write(
            "Cosine similarity identifies listeners with "
            "overlapping music preferences."
        )

        st.write(
            "The top 50 similar listeners contribute their "
            "listening activity to a similarity-weighted "
            "recommendation score."
        )

    with st.expander("03 · Neural Collaborative Filtering (NeuMF)"):

        st.write(
            "NeuMF combines Generalised Matrix Factorisation "
            "(GMF) and a Multi-Layer Perceptron (MLP)."
        )

        st.write(
            "The GMF branch learns element-wise interactions "
            "between user and track embeddings."
        )

        st.write(
            "The MLP branch learns non-linear user-track "
            "relationships through fully connected layers."
        )

        st.write(
            "The two branches are combined to predict "
            "user-track interaction scores. The model is "
            "trained using binary cross-entropy with logits "
            "and negative sampling."
        )

    with st.expander("04 · Hybrid Recommendation"):

        st.write(
            "The hybrid engine combines the ranked outputs of "
            "User-Based Collaborative Filtering, NeuMF and the "
            "popularity baseline."
        )

        st.write(
            "Each model's ranked list is converted into a rank-based "
            "score, then combined using weighted contributions: "
            "60% User-Based CF, 25% NeuMF and 15% popularity."
        )

        st.write(
            "Tracks already present in the target listener's training "
            "history are excluded from the final recommendations."
        )

    st.info(
        "The Create Your Music Profile feature uses a separate "
        "preference-based recommendation approach. It is "
        "implemented but has not undergone independent "
        "quantitative evaluation, so it is excluded from "
        "this comparison."
    )

    st.caption(
        "Metrics are offline ranking results from the held-out "
        "temporal test set. They do not directly measure "
        "subjective listening satisfaction."
    )