import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import os
import altair as alt
import plotly.express as px

from annotated_text import annotated_text



st.set_page_config(layout="wide")

st.markdown("# Recommendation system")

st.markdown("### What is a Recommendation System ?")

st.info("""**Recommendation systems** are AI algorithms built to **suggest** or **recommend** **products** to consumers. 
            They help users discover products and services they might otherwise have not found on their own.""")

st.info("""Recommender systems are built using two different techniques. The first is **Content-based filtering** where recommendations are made based on the user's own preferences.
The second is **Collaborative Filtering** where recommendations are made based on the preferences and behavior of similar users.""")

st.markdown("Here is an example of Content-based filtering versus Collaborative filtering for movie recommendations.")

            
# st.markdown("""Here is an example of **Content-based filtering versus Collaborative filtering** for movie recommendations.""")
st.markdown(" ")

st.image("images/rs.png", width=800)
st.markdown(" ")

st.markdown("""Common applications of Recommendantion systems include:
- **E-Commerce Platforms** 🛍️: Suggest products to users based on their browsing history, purchase patterns, and preferences. 
- **Streaming Services** 📽️: Recommend movies, TV shows, or songs based on users' viewing/listening history and preferences. 
- **Social Media Platforms** 📱: Suggest friends, groups, or content based on users' connections, interests, and engagement history.
- **Automotive and Navigation Systems** 🗺️: Suggest optimal routes based on real-time traffic conditions, historical data, and user preferences.   
""")

api_key = st.secrets["recommendation_system"]["key"]

path_data = r"data/movies"
path_models = r"pretrained_models/recommendation_system"

movies_dict = pickle.load(open(os.path.join(path_data,"movies_dict2.pkl"),"rb"))
movies = pd.DataFrame(movies_dict)

movies.drop_duplicates(inplace=True)


vote_info = pickle.load(open(os.path.join(path_data,"vote_info.pkl"),"rb"))
vote = pd.DataFrame(vote_info)

with open(os.path.join(path_data,'csr_data_tf.pkl'), 'rb') as file:
    csr_data = pickle.load(file)

model = pickle.load(open(os.path.join(path_models,"model.pkl"),"rb"))


def recommend(movie_name, nb):
    n_movies_to_recommend = nb
    idx = movies[movies['title'] == movie_name].index[0]

    distances, indices = model.kneighbors(csr_data[idx], n_neighbors=n_movies_to_recommend + 1)
    idx = list(indices.squeeze())
    df = np.take(movies, idx, axis=0)

    movies_list = list(df.title[1:])

    recommend_movies_names = []
    recommend_posters = []
    movie_ids = []
    for i in movies_list:
        temp_movie_id = (movies[movies.title ==i].movie_id).values[0]
        movie_ids.append(temp_movie_id)
        
        # fetch poster
        recommend_posters.append(fetch_poster(temp_movie_id))
        recommend_movies_names.append(i)
    return recommend_movies_names, recommend_posters, movie_ids


def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

st.divider()

# ''' Frontend '''
st.markdown("  ")
st.markdown("""### Movie Recommendation System 📽️""")
#st.image("images/movies.jpg")
st.markdown("""This example showcases the use of a Recommender System for **movies recommendations** using The Movie Data Base (TMDB). <br>
            The recommender systems uses **Collaborative filtering**, which means it doesn't take into account genre when recommending a movie. <br>
For more info on TMDB, visit https://www.themoviedb.org/ """, unsafe_allow_html=True)
st.markdown(" ")

# st.write(""" <p> Hi, welcome to <b style="color:red">Movies Assistant</b> this free movie recommendation system suggests films based on your interest </p>""", unsafe_allow_html=True)
# st.write("##")


colors = ["#8ef", "#faa", "#afa", "#fea", "#8ef","#afa"]


selected_movie = st.selectbox("**Select a movie**", movies["title"].values[:-3])
selected_nb_movies = st.selectbox("**Select a number of movies to recommend**", np.arange(2,7), index=3)

c1, c2 = st.columns([0.7,0.3], gap="medium")
with c1:
    st.header(selected_movie, divider="grey")
    new_movies = movies.rename({"movie_id":"id"},axis=1).merge(vote, on="id", how="left")
    
    description = new_movies.loc[new_movies["title"]==selected_movie,"description"].to_list()[0]
    genre = new_movies.loc[new_movies["title"]==selected_movie,"genre"].to_list()[0]
    vote_ = new_movies.loc[new_movies["title"]==selected_movie,"vote_average"].to_list()[0]
    vote_count = new_movies.loc[new_movies["title"]==selected_movie,"vote_count"].to_list()[0]
    
    list_genres = [(g.strip(),"",color) for color,g in zip(colors, genre.split(", "))]
    
    st.markdown(f"**Synopsis**: {description}")
    annotated_text(["**Genre(s)**: ", list_genres])
    st.markdown(f"**Rating**: {vote_}:star:")
    st.markdown(f"**Votes**: {vote_count}")

    st.info(f"You've selected {selected_nb_movies} movies to recommend")

    st.markdown(" ")
    recommend_button = st.button("**Recommend movies**")

with c2:
    poster = fetch_poster(movies.loc[movies["title"]==selected_movie,"movie_id"].to_list()[0])
    st.image(poster, width=300)


if recommend_button:
    st.text("Here are few Recommendations..")
    names,posters,movie_ids = recommend(selected_movie, selected_nb_movies)
    
    tab1, tab2 = st.tabs(["View movies", "View genres"])


    with tab1:
        cols=st.columns(int(selected_nb_movies))
        #cols=[col1,col2,col3,col4,col5]
        for i in range(0,selected_nb_movies):
            with cols[i]:
                expander = st.expander("See movie details")
                st.image(posters[i])
                st.markdown(f"##### **{i+1}. {names[i]}**")
                id=movie_ids[i]

                genre = movies.loc[movies["movie_id"]==id,"genre"].to_list()[0]
                list_genres = [(g.strip(),"",color) for color,g in zip(colors, genre.split(", "))]

                synopsis = movies.loc[movies['movie_id']==id, "description"].to_list()[0]
                st.markdown(synopsis)

                vote_avg, vote_count = vote[vote["id"] == id].vote_average , vote[vote["id"] == id].vote_count
                annotated_text(["**Genre(s)**: ", list_genres])
                st.markdown(f"""**Rating**: {list(vote_avg.values)[0]}:star:""")
                st.markdown(f"**Votes**: {list(vote_count.values)[0]}")
    
    
    with tab2:
        recommended_genres = movies.loc[movies["movie_id"].isin(movie_ids[:5]),"genre"].to_list()
        list_recom_genres = [genre for list_genres in recommended_genres for genre in list_genres.split(", ")]
        df_recom_genres = pd.Series(list_recom_genres).value_counts().to_frame().reset_index(names="genre")
        df_recom_genres["proportion (%)"] = (100*df_recom_genres["count"]/df_recom_genres["count"].sum())

        fig = px.bar(df_recom_genres, x='count', y='genre', color="genre", title='Most recommended genres', orientation="h")
        st.plotly_chart(fig, use_container_width=True)


    

