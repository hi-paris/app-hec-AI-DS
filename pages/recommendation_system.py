import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import os
import altair as alt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from annotated_text import annotated_text
from utils import load_data_pickle, load_model_pickle, load_data_csv



st.set_page_config(layout="wide")



st.markdown("# Recommendation system")

st.markdown("### What is a Recommendation System ?")

st.info("""**Recommendation systems** are AI algorithms built to **suggest** or **recommend** **products** to consumers.
        They are very common in social media platforms such as TikTok, Youtube or Instagram or e-commerce websites as they help improve and personalize a consumer's experience.""")

st.markdown("""There are two methods to build recommendation systems:
- **Content-based filtering**: Recommendations are made based on the user's own preferences
- **Collaborative filtering**: Recommendations are made based on the preferences and behavior of similar users""", unsafe_allow_html=True)
            
# st.markdown("""Here is an example of **Content-based filtering versus Collaborative filtering** for movie recommendations.""")
st.markdown(" ")
st.markdown(" ")

_, col_img, _ = st.columns(spec=[0.2,0.6,0.2])
with col_img:
    st.image("images/rs.png")

st.markdown(" ")

st.markdown("""Common applications of Recommendation systems include:
- **E-Commerce Platforms** 🛍️: Suggest products to users based on their browsing history, purchase patterns, and preferences. 
- **Streaming Services** 📽️: Recommend movies, TV shows, or songs based on users' viewing/listening history and preferences. 
- **Social Media Platforms** 📱: Suggest friends, groups, or content based on users' connections, interests, and engagement history.
- **Automotive and Navigation Systems** 🗺️: Suggest optimal routes based on real-time traffic conditions, historical data, and user preferences.   
""")

st.markdown(" ")

select_usecase = st.selectbox("**Choose a use case**", 
                              ["Movie recommendation system 📽️", 
                               "Hotel recommendation system 🛎️"])

st.divider()



#####################################################################################################
#                                       MOVIE RECOMMENDATION SYSTEM                                 #
#####################################################################################################

# Recommendation function
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
        try:
            poster = fetch_poster(temp_movie_id)
            recommend_posters.append(poster)
        except:
            recommend_posters.append(None)
        
        recommend_movies_names.append(i)
    return recommend_movies_names, recommend_posters, movie_ids

# Get poster
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]



if select_usecase == "Movie recommendation system 📽️":

    colors = ["#8ef", "#faa", "#afa", "#fea", "#8ef","#afa"]
    api_key = st.secrets["recommendation_system"]["key"]

    # Load data 
    path_data = r"data/movies"
    path_models = r"pretrained_models/recommendation_system"

    movies_dict = pickle.load(open(os.path.join(path_data,"movies_dict2.pkl"),"rb"))
    movies = pd.DataFrame(movies_dict)
    movies.drop_duplicates(inplace=True)

    vote_info = pickle.load(open(os.path.join(path_data,"vote_info.pkl"),"rb"))
    vote = pd.DataFrame(vote_info)

    # Load model
    model = load_model_pickle(path_models,"model.pkl")
    with open(os.path.join(path_data,'csr_data_tf.pkl'), 'rb') as file:
        csr_data = pickle.load(file)


    # Description of the use case
    st.markdown("""## Movie Recommendation System 📽️""")

    #st.info(""" """)

    st.markdown("""This use case showcases the use of recommender systems for **movie recommendations** using **collaborative filtering**. <br>
                   The model recommends and ranks movies based on what users, who have also watched the chosen movie, have watched else on the platform. <br> 
    """, unsafe_allow_html=True)
    st.markdown(" ")

    
    # User selection
    selected_movie = st.selectbox("**Select a movie**", movies["title"].values[:-3])
    selected_nb_movies = st.selectbox("**Select a number of movies to recommend**", np.arange(2,7), index=3)

    # Show user selection on the app
    c1, c2 = st.columns([0.7,0.3], gap="medium")
    with c1:
        new_movies = movies.rename({"movie_id":"id"},axis=1).merge(vote, on="id", how="left")
        description = new_movies.loc[new_movies["title"]==selected_movie,"description"].to_list()[0]
        genre = new_movies.loc[new_movies["title"]==selected_movie,"genre"].to_list()[0]
        vote_ = new_movies.loc[new_movies["title"]==selected_movie,"vote_average"].to_list()[0]
        vote_count = new_movies.loc[new_movies["title"]==selected_movie,"vote_count"].to_list()[0]
        
        list_genres = [(g.strip(),"",color) for color,g in zip(colors, genre.split(", "))]
        
        st.header(selected_movie, divider="grey")
        st.markdown(f"**Synopsis**: {description}")
        annotated_text(["**Genre(s)**: ", list_genres])
        st.markdown(f"**Rating**: {vote_}:star:")
        st.markdown(f"**Votes**: {vote_count}")

        st.info(f"You've selected {selected_nb_movies} movies to recommend")
        st.markdown(" ")
        
        recommend_button = st.button("**Recommend movies**")

    with c2:
        try:
            poster = fetch_poster(movies.loc[movies["title"]==selected_movie,"movie_id"].to_list()[0])
            st.image(poster, width=300)
        except:
            pass


    # Run model and show results
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
                    
                    if posters[i] == None:
                        pass
                    else:
                        st.image(posters[i])
                    
                    st.markdown(f"##### **{i+1}. {names[i]}**")
                    id = movie_ids[i]

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







#####################################################################################################
#                                       HOTEL RECOMMENDATION SYSTEM                                 #
#####################################################################################################
            

# Load scaler with caching
    
    @st.cache_data()
    def get_scaler(df):
        scaler = MinMaxScaler()
        scaler.fit(df[['Rating', 'Price']])
        return scaler

    def recommend_hotels_with_location_and_beds(df, preferences, max_recommendations=5):
        # Start with the full dataset
        filtered_df = df.copy()
        
        # Filter by Location if specified (either city or country)
        if 'Location' in preferences and preferences['Location']:
            filtered_df = filtered_df[(filtered_df['City'].str.contains(preferences['Location'], case=False, na=False)) |
                                    (filtered_df['Country'].str.contains(preferences['Location'], case=False, na=False))]

        # Filter by Number of beds if specified
        if 'Number of beds' in preferences:
            filtered_df = filtered_df[filtered_df['Number of bed'] == preferences['Number of beds']]
        
        # Filter by Rating if specified
        if 'Rating' in preferences:
            min_rating, max_rating = preferences['Rating']
            filtered_df = filtered_df[filtered_df['Rating'].between(min_rating, max_rating)]
        
        # Filter by Price range if specified
        if 'Price' in preferences:
            min_price, max_price = preferences['Price']
            filtered_df = filtered_df[filtered_df['Price'].between(min_price, max_price)]

        # Ensure there are still hotels after filtering
        if filtered_df.empty:
            # Send a notification if no hotels match the criteria
            send_notification("No hotels were found matching the specified criteria.")
            return pd.DataFrame(), "No hotels were found matching the specified criteria."
        
        preferences["Rating"] = np.mean(np.array(preferences["Rating"]))
        preferences["Price"] = np.mean(np.array(preferences["Price"]))

        # Normalize the preferences vector (excluding location and number of beds for similarity calculation)
        preferences_vector = np.array([[preferences.get('Rating', 0),
                                        preferences.get('Price', 0)]])
        preferences_vector_normalized = scaler.transform(preferences_vector)

        # Calculate similarity scores for the filtered hotels
        filtered_numerical_features = filtered_df[['Rating', 'Price']]
        filtered_numerical_features_normalized = scaler.transform(filtered_numerical_features)
        similarity_scores = cosine_similarity(preferences_vector_normalized, filtered_numerical_features_normalized)[0]

        # Get the indices of top_n similar hotels
        top_indices = similarity_scores.argsort()[-max_recommendations:][::-1]
        recommended_indices = filtered_df.iloc[top_indices].index

        # Return the recommended hotels with relevant details (including specified columns)
        return df.loc[recommended_indices], None


    def send_notification(message):
        """
        Placeholder function to send a notification.
        This function can be replaced with the actual notification mechanism (e.g., email, SMS).
        """
        print("Notification:", message)


    def country_info(country):
        if country == "Thailand":
            image = "images/thailand.jpeg"
            emoji = "🏝️"
            description = """**Description**: 
Thailand seamlessly fuses ancient traditions with modern dynamism, creating an unparalleled tapestry for travelers. 
Renowned for its warm hospitality, vibrant culture, and delectable cuisine, Thailand offers an unforgettable experience for every adventurer."""
            top_places = """
- **Bangkok**: Immerse yourself in the hustle and bustle of Bangkok's streets, adorned with glittering temples and bustling markets. The Grand Palace and Khao San Road showcase the city's unique blend of tradition and modernity.
- **Chiang Mai**: Nestled in the misty mountains of Northern Thailand, Chiang Mai captivates with ancient temples, lush landscapes, and vibrant night markets. The Old City exudes a unique atmosphere, while the surrounding hills offer tranquility.
- **Phuket**: Thailand's largest island, Phuket, beckons beach lovers with its stunning white sands, vibrant nightlife, and water activities. It's a perfect blend of relaxation and excitement."""

        if country == "France":
            image = "images/france.jpeg"
            emoji = "⚜️"
            description ="""**Description**:
Indulge in the countries rich tapestry of art, culture, and gastronomy. 
From the romantic allure of Paris to the sun-kissed vineyards of Provence, every corner of this diverse country tells a unique story, promising an unforgettable journey for every traveler."""
            top_places = """ 
- **Paris**: Dive into the city's iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral grace the skyline.
- **Provence**: Visit the stunning Palais des Papes in Avignon, explore the colorful markets of Aix-en-Provence, and unwind in the serene beauty of the Luberon region.
- **Côte d'Azur**: This stunning stretch of the French coastline is a captivating blend of azure waters, picturesque landscapes and charming villages.
"""

        if country == "Spain":
            image = "images/spain-banner.jpg"
            emoji = "☀️"
            description = """**Description**:
Embark on an unforgettable journey where tradition and modernity coexist in harmony. 
From the lively streets of Barcelona to the sun-soaked beaches of Andalusia, Spain offers a captivating blend of history, culture, and natural beauty.            
"""
            top_places = """
- **Barcelona**: Explore the iconic Sagrada Familia, stroll down the vibrant La Rambla, and soak in the Mediterranean ambiance at Barceloneta Beach.
- **Seville**: Visit the awe-inspiring Alcázar, marvel at the Giralda Tower, and wander through the enchanting alleys of the Santa Cruz neighborhood.
- **Granada**: Explore the Generalife Gardens, stroll through the Albayzín quarter with its narrow streets and white houses, and savor the views of the city from the Mirador de San Nicolás.
"""

        if country == "Singapore":
            image = "images/singapore.jpg"
            emoji = "🏙️"
            description = """**Description**:
From gleaming skyscrapers to vibrant neighborhoods, this cosmopolitan gem in Southeast Asia promises an immersive journey into a world where tradition meets cutting-edge technology."""

            top_places = """
- **Marina Bay Sands**: Enjoy panoramic views from the SkyPark, take a dip in the infinity pool, and explore The Shoppes for luxury shopping and entertainment. At night, witness the mesmerizing light and water show at the Marina Bay Sands Skypark.
- **Gardens by the Bay**: Explore the Flower Dome and Cloud Forest conservatories, and stroll through the scenic OCBC Skyway for breathtaking views of the gardens and city.
- **Sentosa Island**: Escape to Sentosa Island, a resort destination offering a myriad of attractions. Relax on pristine beaches, visit Universal Studios Singapore for thrilling rides, and explore S.E.A. Aquarium for an underwater adventure.

"""

        ###### STREAMLIT MARKDOWN ######
        st.header(f"{country} {emoji}", divider="grey")
        st.image(image)
        st.markdown(description)

        see_top_places = st.checkbox("**Top places to visit**", key={country})
        if see_top_places:
            st.markdown(top_places)
        


if select_usecase == "Hotel recommendation system 🛎️":

    st.markdown("""## Hotel Recommendation System 🛎️""")

    st.info("""This use case shows how you can create personalized hotel recommendations using a recommendation system with **content-based Filtering**. 
                Analyzing location, amenities, price, and reviews, the model suggests tailored hotel recommendation based on the user's preference.
    """)
    st.markdown(" ")


    path_hotels_data = r"data/hotels"

    # Load hotel data
    df = load_data_csv(path_hotels_data,"booking_df.csv")

    # clean data
    df.drop_duplicates(inplace=True)
    df["Country"] = df["Country"].apply(lambda x: "Spain" if x=="Espagne" else x)
    list_cities = df["City"].value_counts().to_frame().reset_index()
    list_cities = list_cities.loc[list_cities["count"]>=5,"City"].to_numpy()
    df = df.loc[(df["City"].isin(list_cities)) & (df["Number of bed"]<=6)]
    df["Price"] = df["Price"].astype(int)
    df.loc[(df["Number of bed"]==0) & (df["Price"]<1000),"Number of bed"] = 1
    df.loc[(df["Number of bed"]==0) & (df["Price"].between(1000,2000)),"Number of bed"] = 2
    df.loc[(df["Number of bed"]==0) & (df["Price"]>2000),"Number of bed"] = 3

    df["Rating"] = df["Rating"].apply(lambda x: np.nan if x==0 else x)
    df["Rating"].fillna(np.round(df["Rating"].mean(), 1), inplace=True)

    scaler = get_scaler(df)


    col1, col2 = st.columns([0.3,0.7], gap="large")

    with col1:
        # Collect user preferences
        st.markdown(" ")
        st.markdown(" ")
        st.markdown("")
        #st.markdown("#### Filter preferences")
        list_countries = df["Country"].unique()
        location = st.selectbox("Select a Country",list_countries, index=0)

        list_nb_beds = df["Number of bed"].unique()
        num_beds = st.selectbox("Number of beds", list_nb_beds, index=0)
        #if num_beds == "No information"

        min_rating, max_rating = st.slider("Range of ratings", min_value=df["Rating"].min(), max_value=df["Rating"].max(), step=0.1, value=(5.0, df["Rating"].max()))
        min_price, max_price = st.slider("Range of room prices", min_value=df["Price"].min(), max_value=df["Price"].max(), step=10, value=(df["Price"].min(), 10000))

        # Convert price range sliders to integer values
        min_price = int(min_price)
        max_price = int(max_price)

    with col2:
        country_info(location)


    preferences = {
        'Location': location,
        'Number of beds': num_beds,
        'Rating': [min_rating, max_rating],
        'Price': [min_price, max_price],
    }
            

    if st.button("Recommend Hotels"):
        st.info("Hotels were recommended based on how similar they were to the users preferences.")
        
        # Default number of recommendations to show
        max_recommendations = 5
        
        # Call the recommendation function
        recommended_hotels, message = recommend_hotels_with_location_and_beds(df, preferences, max_recommendations)
        
        # If no recommendations, reduce the maximum number of recommendations and try again
        if recommended_hotels.empty:
            max_recommendations -= 1
            recommended_hotels, message = recommend_hotels_with_location_and_beds(df, preferences, max_recommendations)
            if recommended_hotels.empty:
                st.error(message)
            # else:
            #     st.write(recommended_hotels)
        else:
            st.markdown(" ")
            for i in range(len(recommended_hotels)):
                #st.dataframe(recommended_hotels)
                df_result = recommended_hotels.iloc[i,:]                    
                col1_, col2_ = st.columns([0.4,0.6], gap="medium")

                with col1_:
                    st.image("images/room.jpg",width=100)
                    st.markdown(f"### {i+1}: {df_result['Hotel Name']}")
                    st.markdown(f"""**{df_result['Room Type']}** <br>
                                with {df_result['Bed Type']}
                                """, unsafe_allow_html=True)
                with col2_:
                    st.markdown(" ")
                    st.markdown(" ")
                    annotated_text("**Number of beds :** ",(f"{df_result['Number of bed']}","","#faa"))
                    #st.markdown(f"**Bed type**: {df_result['Bed Type']}")
                    annotated_text("**City:** ",(f"{df_result['City']}","","#afa"))
                    annotated_text("**Rating:** ",(f"{df_result['Rating']}","","#8ef"))
                    annotated_text("**Price:** ",(f"{df_result['Price']}$","","#fea"))
                
                st.divider()
                    