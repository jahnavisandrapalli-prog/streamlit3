import ast
import io
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="IMDb Movie Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_movies(file_bytes: bytes) -> pd.DataFrame:
    movies = pd.read_csv(io.BytesIO(file_bytes))
    required_columns = {
        "id",
        "primaryTitle",
        "averageRating",
        "numVotes",
        "genres",
        "countriesOfOrigin",
        "grossWorldwide",
    }
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in ("averageRating", "numVotes", "grossWorldwide"):
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

    def parse_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None or pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            parsed = str(value).split(",")
        if not isinstance(parsed, list):
            parsed = [parsed]
        return [str(item).strip() for item in parsed if str(item).strip()]

    movies["genres"] = movies["genres"].map(parse_list)
    movies["countriesOfOrigin"] = movies["countriesOfOrigin"].map(parse_list)
    return movies


def render_dataset_information(movies: pd.DataFrame) -> None:
    with st.expander("Dataset Information", expanded=False):
        st.markdown(
            f"**{len(movies):,} movies** across **{len(movies.columns)} columns**."
        )
        st.write("**Columns:**", ", ".join(movies.columns))
        st.write("**Column data types**")
        st.dataframe(
            movies.dtypes.astype(str).rename("Data type").to_frame(),
            width="stretch",
        )
        st.write("**Missing values by column**")
        missing_values = movies.isna().sum().rename("Missing values").to_frame()
        st.dataframe(missing_values, width="stretch")
        st.write("**First 10 records**")
        st.dataframe(movies.head(10), width="stretch", hide_index=True)


def render_charts(movies: pd.DataFrame) -> None:
    genre_counts = (
        movies["genres"]
        .explode()
        .dropna()
        .value_counts()
        .rename_axis("Genre")
        .reset_index(name="Movies")
    )
    ratings = movies.dropna(subset=["averageRating"])
    gross_movies = (
        movies.dropna(subset=["grossWorldwide"])
        .nlargest(10, "grossWorldwide")
        .sort_values("grossWorldwide", ascending=True)
    )

    genre_chart = px.bar(
        genre_counts,
        x="Genre",
        y="Movies",
        text="Movies",
        color="Genre",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    genre_chart.update_layout(showlegend=False, xaxis_title="Genre")
    histogram = px.histogram(
        ratings,
        x="averageRating",
        nbins=15,
        labels={"averageRating": "IMDb Rating", "count": "Movies"},
        color_discrete_sequence=["#2f80ed"],
    )
    scatter = px.scatter(
        movies.dropna(subset=["averageRating", "numVotes"]),
        x="averageRating",
        y="numVotes",
        hover_name="primaryTitle",
        log_y=True,
        labels={
            "averageRating": "IMDb Rating",
            "numVotes": "Number of Votes (log scale)",
        },
        color_discrete_sequence=["#168cff"],
    )
    gross_chart = px.bar(
        gross_movies,
        x="grossWorldwide",
        y="primaryTitle",
        orientation="h",
        labels={
            "grossWorldwide": "Worldwide Gross (USD)",
            "primaryTitle": "Movie",
        },
        color="grossWorldwide",
        color_continuous_scale="Oranges",
    )
    gross_chart.update_xaxes(tickformat="$.2s")

    for chart in (genre_chart, histogram, scatter, gross_chart):
        chart.update_layout(
            height=190,
            margin=dict(l=12, r=12, t=8, b=24),
            paper_bgcolor="#f5f7fb",
            plot_bgcolor="#f5f7fb",
            font=dict(color="#243247", size=9),
            coloraxis_showscale=False,
        )
        chart.update_xaxes(
            gridcolor="#dce3ed",
            zerolinecolor="#dce3ed",
            title_font_size=9,
            tickfont_size=8,
        )
        chart.update_yaxes(
            gridcolor="#dce3ed",
            zerolinecolor="#dce3ed",
            title_font_size=9,
            tickfont_size=8,
        )

    def render_chart_row(
        section: str, chart_cards: tuple[tuple[str, object], ...]
    ) -> None:
        columns = st.columns(2, gap="small")
        for column, (title, chart) in zip(columns, chart_cards):
            with column:
                with st.container(border=True):
                    st.caption(title)
                    st.plotly_chart(
                        chart,
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"{section}-{title}",
                    )

    overview_tab, rating_tab, gross_tab = st.tabs(
        ["📊 Overview", "⭐ Rating Analysis", "💰 Gross Collection"]
    )
    with overview_tab:
        render_chart_row(
            "overview-first-row",
            (("Movies by Genre", genre_chart), ("IMDb Rating Distribution", histogram))
        )
        render_chart_row(
            "overview-second-row",
            (
                ("Rating vs. Number of Votes", scatter),
                ("Top 10 Movies by Worldwide Gross", gross_chart),
            )
        )
    with rating_tab:
        render_chart_row(
            "rating",
            (
                ("IMDb Rating Distribution", histogram),
                ("Rating vs. Number of Votes", scatter),
            )
        )

    with gross_tab:
        with st.container(border=True):
            st.caption("Top 10 Movies by Worldwide Gross")
            st.plotly_chart(
                gross_chart,
                width="stretch",
                config={"displayModeBar": False},
                key="gross-detail",
            )


def reset_explorer() -> None:
    for key in (
        "explorer_genre",
        "explorer_countries",
        "explorer_rating",
        "explorer_votes",
        "explorer_title",
        "explorer_sort",
    ):
        st.session_state.pop(key, None)


def render_movie_explorer(movies: pd.DataFrame) -> None:
    title_col, reset_col = st.columns([4, 1])
    title_col.subheader("🎞️ Interactive Movie Explorer")
    reset_col.button(
        "↺ Reset",
        on_click=reset_explorer,
        key="reset_explorer",
        width="stretch",
    )
    genres = sorted({genre for movie_genres in movies["genres"] for genre in movie_genres})
    countries = sorted(
        {
            country
            for movie_countries in movies["countriesOfOrigin"]
            for country in movie_countries
        }
    )
    ratings = movies["averageRating"].dropna()
    rating_bounds = (float(ratings.min()), float(ratings.max()))
    max_votes = int(movies["numVotes"].max()) if movies["numVotes"].notna().any() else 0

    (
        genre_col,
        country_col,
        rating_col,
        votes_col,
        title_search_col,
        sort_col,
    ) = st.columns([1, 1.2, 1.2, 0.9, 1.2, 1])
    with genre_col:
        selected_genre = st.selectbox(
            "Genre",
            ["All genres", *genres],
            key="explorer_genre",
        )
    with country_col:
        selected_countries = st.multiselect(
            "Countries of origin",
            countries,
            key="explorer_countries",
        )
    with rating_col:
        rating_range = st.slider(
            "Rating",
            min_value=rating_bounds[0],
            max_value=rating_bounds[1],
            value=rating_bounds,
            step=0.1,
            key="explorer_rating",
        )
    with votes_col:
        minimum_votes = st.number_input(
            "Min votes",
            min_value=0,
            max_value=max_votes,
            value=min(100_000, max_votes),
            step=10_000,
            key="explorer_votes",
        )
    with title_search_col:
        title_query = st.text_input("Search title", key="explorer_title")
    with sort_col:
        sort_order = st.radio(
            "Sort order",
            ["Descending", "Ascending"],
            horizontal=True,
            key="explorer_sort",
        )

    filtered = movies.copy()
    if selected_genre != "All genres":
        filtered = filtered[
            filtered["genres"].map(lambda movie_genres: selected_genre in movie_genres)
        ]
    if selected_countries:
        filtered = filtered[
            filtered["countriesOfOrigin"].map(
                lambda movie_countries: any(
                    country in movie_countries for country in selected_countries
                )
            )
        ]
    filtered = filtered[
        filtered["averageRating"].between(*rating_range)
        & filtered["numVotes"].ge(minimum_votes)
    ]
    if title_query.strip():
        filtered = filtered[
            filtered["primaryTitle"].str.contains(
                title_query.strip(), case=False, na=False, regex=False
            )
        ]
    filtered = filtered.sort_values(
        "averageRating", ascending=sort_order == "Ascending"
    )

    st.caption(f"Showing **{len(filtered):,}** of {len(movies):,} movies")
    display_columns = [
        column
        for column in (
            "primaryTitle",
            "genres",
            "countriesOfOrigin",
            "averageRating",
            "numVotes",
            "grossWorldwide",
        )
        if column in filtered.columns
    ]
    st.dataframe(
        filtered[display_columns],
        width="stretch",
        height=240,
        hide_index=True,
        column_config={
            "primaryTitle": st.column_config.TextColumn("Title"),
            "genres": st.column_config.ListColumn("Genre"),
            "countriesOfOrigin": st.column_config.ListColumn("Country"),
            "averageRating": st.column_config.NumberColumn("IMDb Rating", format="%.1f"),
            "numVotes": st.column_config.NumberColumn("Votes", format="%d"),
            "grossWorldwide": st.column_config.NumberColumn(
                "Worldwide Gross", format="$%,.0f"
            ),
        },
    )


def render_movie_search(movies: pd.DataFrame) -> None:
    st.subheader("🔎 Movie Search")
    genres = sorted({genre for movie_genres in movies["genres"] for genre in movie_genres})
    if "selected_movie_id" not in st.session_state:
        st.session_state.selected_movie_id = None

    with st.form("movie_search_form"):
        title_query = st.text_input("Movie title", key="form_movie_title")
        preferred_genre = st.selectbox(
            "Preferred genre",
            ["Any genre", *genres],
            key="form_movie_genre",
        )
        minimum_rating = st.slider(
            "Minimum IMDb rating",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            key="form_minimum_rating",
        )
        submitted = st.form_submit_button("Search Movie", type="primary")

    if submitted:
        matches = movies[movies["averageRating"].ge(minimum_rating)]
        if preferred_genre != "Any genre":
            matches = matches[
                matches["genres"].map(
                    lambda movie_genres: preferred_genre in movie_genres
                )
            ]
        if title_query.strip():
            matches = matches[
                matches["primaryTitle"].str.contains(
                    title_query.strip(), case=False, na=False, regex=False
                )
            ]
        matches = matches.sort_values("averageRating", ascending=False)
        st.session_state.selected_movie_id = (
            matches.iloc[0]["id"] if not matches.empty else None
        )
        if matches.empty:
            st.info("No movies match those search preferences. Try broadening your search.")
        else:
            st.success(f"Found a match: {matches.iloc[0]['primaryTitle']}")

    selected_id = st.session_state.selected_movie_id
    if selected_id is not None:
        selected = movies.loc[movies["id"].eq(selected_id)]
        if not selected.empty:
            movie = selected.iloc[0]
            with st.container(border=True):
                image_url = movie.get("primaryImage")
                if isinstance(image_url, str) and image_url.startswith("https://"):
                    st.image(image_url, width="stretch")
                st.markdown(f"#### {movie['primaryTitle']}")
                st.caption(", ".join(movie["genres"]) or "Genre not available")
                rating_col, votes_col = st.columns(2)
                rating_col.metric("IMDb Rating", f"{movie['averageRating']:.1f}")
                votes_col.metric("Votes", f"{int(movie['numVotes']):,}")


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #101725; color: #eef3fb; }
        [data-testid="stSidebar"] { background: #111a2a; }
        .block-container {
            max-width: 1500px;
            padding: .55rem 1rem .7rem;
        }
        [data-testid="stVerticalBlock"] { gap: .55rem; }
        [data-testid="stHorizontalBlock"] {
            gap: .55rem;
            align-items: stretch;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1d5eaa, #277dd0);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 9px;
            min-height: 66px;
            padding: 8px 11px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: white; }
        [data-testid="stMetricLabel"] { font-size: .72rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2)
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #16865d, #22ae74);
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3)
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #6734a8, #8650cf);
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4)
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #d65b0b, #fa8318);
        }
        [data-baseweb="tab-list"] { gap: .25rem; }
        [data-baseweb="tab"] {
            height: 2rem;
            padding: 0 .7rem;
            font-size: .8rem;
        }
        [data-testid="stPlotlyChart"] { margin-bottom: -.8rem; }
        [data-testid="stDataFrame"] { font-size: .78rem; }
        [data-testid="stSidebar"] h2 { font-size: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    dataset_path = Path(__file__).with_name("movies.csv")
    with st.sidebar:
        st.markdown("## 🎬 IMDb Movie Analytics")
        st.caption("Dashboard · Explore · Discover")
        page = st.radio(
            "Navigation",
            ["Dashboard", "Movie Explorer", "Movie Search"],
            label_visibility="collapsed",
        )
        with st.expander("Upload another CSV"):
            uploaded_file = st.file_uploader(
                "Choose a movie dataset",
                type="csv",
                help="The included Top 250 dataset is used unless you upload another CSV.",
            )
        st.divider()
        st.markdown("**Dataset at a glance**")

    if uploaded_file is not None:
        data_bytes = uploaded_file.getvalue()
    elif dataset_path.exists():
        data_bytes = dataset_path.read_bytes()
    else:
        st.error(f"Could not find the default dataset: {dataset_path}")
        st.stop()

    try:
        movies = load_movies(data_bytes)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Unable to load the movie dataset: {error}")
        st.stop()

    with st.sidebar:
        st.write(f"Rows: **{len(movies):,}**")
        st.write(f"Columns: **{len(movies.columns)}**")
        st.caption("Includes movie titles, genres, ratings, votes, and worldwide gross.")
        st.markdown(
            """
            <div style="text-align: right; margin-top: 1.5rem; opacity: 0.75;
                        font-size: 0.75rem; line-height: 1.45;">
                <strong>MokshithaReddy</strong><br>
                PRN: 250200219
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page == "Dashboard":
        banner_path = Path(__file__).with_name("dashboard-banner.jpg")
        st.image(str(banner_path), width="stretch")
        total_votes = movies["numVotes"].sum()
        average_rating = movies["averageRating"].mean()
        highest_rating = movies["averageRating"].max()
        metrics = st.columns(4)
        metrics[0].metric("🎬 Total Movies", f"{len(movies):,}")
        metrics[1].metric(
            "⭐ Average IMDb Rating",
            f"{average_rating:.1f}" if not math.isnan(average_rating) else "N/A",
        )
        metrics[2].metric(
            "🏆 Highest Rating",
            f"{highest_rating:.1f}" if not math.isnan(highest_rating) else "N/A",
        )
        metrics[3].metric("👥 Total Votes", f"{total_votes:,.0f}")

        st.divider()
        render_charts(movies)
        st.success("Dataset loaded successfully! Enjoy exploring the IMDb Top 250.")
        render_dataset_information(movies)
        st.divider()
        explorer_col, search_col = st.columns([1.8, 1])
        with explorer_col:
            render_movie_explorer(movies)
        with search_col:
            render_movie_search(movies)
    elif page == "Movie Explorer":
        st.title("🎞️ Interactive Movie Explorer")
        st.caption("Filter and sort the IMDb Top 250 using all controls below.")
        render_movie_explorer(movies)
    else:
        st.title("🔎 Movie Search")
        st.caption("Save a movie preference and search across Streamlit interactions.")
        render_movie_search(movies)


if __name__ == "__main__":
    main()
