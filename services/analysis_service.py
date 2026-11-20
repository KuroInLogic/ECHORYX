from collections import Counter
from services.data_service import load_music_data

def analyze_listening_data():
    artist_plays = Counter()
    track_plays = Counter()
    user_plays = Counter()

    for chunk in load_music_data():

        artist_plays.update(
            chunk["artist_name"].dropna()
        )

        track_plays.update(
            zip(
                chunk["artist_name"],
                chunk["track_name"]
            )
        )

        user_plays.update(
            chunk["user_id"].dropna()
        )

    return {
        "top_artists": artist_plays.most_common(10),
        "top_tracks": track_plays.most_common(10),
        "top_users": user_plays.most_common(10),
        "total_users": len(user_plays),
        "average_plays_per_user": (
            sum(user_plays.values()) / len(user_plays)
        )
    }
