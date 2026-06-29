import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# set page Configuration
st.set_page_config(
    layout="wide",
    page_title="MovieMatrix",
    page_icon="Red Black Typography Nine Brand Logo (2).png"
)
# read dataset
df = pd.read_csv("movies_dataset.csv")

# create option menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Dataset", "Prediction", "About"],
                           icons=["house", "table", "robot", "info-circle"],
                           default_index=0,
                           menu_icon="cast")
if selected == "Home":

    st.title("MovieMatrix")
    st.subheader("Intelligent Film Matching System")

    st.write("""
Welcome to **MovieMatrix Pro**, an AI-powered movie recommendation system
that predicts whether a user is likely to enjoy a movie based on user
preferences, ratings, genres, and viewing behavior.
""")
    st.divider()

#Metrics

    total_records = len(df)
    total_movies = df["Movie_Title"].nunique()
    total_genres = df["Movie_Genre"].nunique()
    avg_rating = round(df["User_Rating"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(" Total Records", total_records)
    col2.metric(" Total Movies", total_movies)
    col3.metric(" Genres", total_genres)
    col4.metric(" Avg Rating", avg_rating)

    st.divider()
    st.header("Dashboard Overview")

# Row 1
    fig1, fig2 = st.columns(2)
    with fig1:
        st.subheader("Movie Genre Distribution")
        genre = df["Movie_Genre"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(genre.index, genre.values)
        ax.set_xlabel("Genre")
        ax.set_ylabel("Movies")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with fig2:

        st.subheader("Like vs Dislike")
        liked = df["Liked"].value_counts()
        labels = ["Liked", "Disliked"]
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            liked.values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
    st.divider()

#Row 2]
    fig3, fig4 = st.columns(2)

    with fig3:
        st.subheader("Average Rating by Genre")
        rating = df.groupby("Movie_Genre")["User_Rating"].mean()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(rating.index, rating.values)
        ax.set_xlabel("Genre")
        ax.set_ylabel("Average Rating")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with fig4:
        st.subheader("User Gender Distribution")
        gender = df["User_Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            gender.values,
            labels=gender.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
    st.divider()

elif selected == "Dataset":

    st.title("Dataset Overview")
    st.write("Explore the movie dataset used to train the Machine Learning model.")
    st.divider()

#Dataset Metrics
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    total_movies = df["Movie_Title"].nunique()
    total_genres = df["Movie_Genre"].nunique()

    col1, col2 = st.columns(2)

    col1.metric("📄 Rows", total_rows)
    col2.metric("📑 Columns", total_columns)
    st.divider()

#Dataset Preview
    st.subheader(" Dataset Preview")
    st.dataframe(df, use_container_width=True)
    st.divider()

    st.divider()

#Dataset Information
    st.subheader("Dataset Summary")

    st.success(f"""
✔ Total Records : {total_rows}

✔ Total Movies : {total_movies}

✔ Total Genres : {total_genres}

✔ Target Variable : Liked

✔ Machine Learning Task : Classification
""")

elif selected == "Prediction":

    st.title("Movie Like Prediction")
    st.write("Enter the movie and user details to predict whether the user will like the movie.")
    st.divider()

#values
    movie_title_list = sorted(df["Movie_Title"].unique())
    movie_genre_list = sorted(df["Movie_Genre"].unique())
    gender_list = sorted(df["User_Gender"].unique())
    preferred_list = sorted(df["Preferred_Genre"].unique())
    time_list = sorted(df["Time_Of_Day"].unique())

#label encoding
    le_title = LabelEncoder()
    le_genre = LabelEncoder()
    le_gender = LabelEncoder()
    le_pref = LabelEncoder()
    le_time = LabelEncoder()

    df["Movie_Title"] = le_title.fit_transform(df["Movie_Title"])
    df["Movie_Genre"] = le_genre.fit_transform(df["Movie_Genre"])
    df["User_Gender"] = le_gender.fit_transform(df["User_Gender"])
    df["Preferred_Genre"] = le_pref.fit_transform(df["Preferred_Genre"])
    df["Time_Of_Day"] = le_time.fit_transform(df["Time_Of_Day"])

#features
    x = df[[
        "Movie_Title",
        "Movie_Genre",
        "Release_Year",
        "Movie_Duration_Min",
        "User_Age",
        "User_Gender",
        "Preferred_Genre",
        "Time_Of_Day",
        "User_Rating",
        "Watched_Duration_Pct"
    ]]
    y = df["Liked"]

#train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=0.80,
        random_state=42
    )
#model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    #accuracy = accuracy_score(y_test, pred)
    #st.metric("Model Accuracy", f"{accuracy:.2%}")


#user input
    st.subheader("Movie Details")
    left, right = st.columns(2)
    with left:
        movie_title = st.selectbox(
            "Movie Title",
            movie_title_list
        )
        movie_genre = st.selectbox(
            "Movie Genre",
            movie_genre_list
        )
        release_year = st.number_input(
            "Release Year",
            min_value=1980,
            max_value=2035,
            value=2023
        )
        duration = st.number_input(
            "Movie Duration (Minutes)",
            min_value=30,
            max_value=300,
            value=120
        )
        rating = st.slider(
            "User Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )
    with right:
        user_age = st.number_input(
            "User Age",
            min_value=10,
            max_value=100,
            value=25
        )
        gender = st.selectbox(
            "User Gender",
            gender_list
        )
        preferred = st.selectbox(
            "Preferred Genre",
            preferred_list
        )
        time = st.selectbox(
            "Time Of Day",
            time_list
        )
        watched = st.slider(
            "Watched Duration (%)",
            min_value=0,
            max_value=100,
            value=80
        )
    st.divider()

    if st.button("🎬 Predict Movie Preference"):
        new_data = pd.DataFrame({
            "Movie_Title":[le_title.transform([movie_title])[0]],
            "Movie_Genre":[le_genre.transform([movie_genre])[0]],
            "Release_Year":[release_year],
            "Movie_Duration_Min":[duration],
            "User_Age":[user_age],
            "User_Gender":[le_gender.transform([gender])[0]],
            "Preferred_Genre":[le_pref.transform([preferred])[0]],
            "Time_Of_Day":[le_time.transform([time])[0]],
            "User_Rating":[rating],
            "Watched_Duration_Pct":[watched]
        })

#prediction
        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)
        #confidence = probability.max() * 100
        st.divider()
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                st.success("✅ User is likely to Like this Movie")
            else:
                st.error("❌ User is unlikely to Like this Movie")
        st.divider()

       
        #st.success(f"Model Accuracy : {accuracy:.2%}")

elif selected == "About":
    st.title("About MovieMatrix")
    st.write("Learn more about the project, dataset, machine learning model, and technologies used.")
    st.divider()

#project description
    st.header("Project Description")
    st.write("""
MovieMatrix Pro is an AI and Machine Learning based movie recommendation
system that predicts whether a user is likely to enjoy a movie.

The prediction is based on various user preferences such as movie genre,
release year, user age, preferred genre, viewing time, movie rating,
and watched duration percentage.

The project demonstrates the practical implementation of Machine Learning
for intelligent movie recommendation.
""")
    st.divider()

#project objetive
    st.header("Project Objective")
    st.write("""
• Predict whether a user will like a movie.

• Analyze movie and user behaviour.

• Build a Machine Learning based recommendation system.

• Improve movie recommendation accuracy.

• Provide a simple and interactive user interface.
""")
    st.divider()

# #dataset
#     st.header("Dataset Information")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Total Records", len(df))
#         st.metric("Movies", df["Movie_Title"].nunique())
#         st.metric("Genres", df["Movie_Genre"].nunique())
#     with col2:
#         st.metric("Features", len(df.columns)-1)
#         st.metric("Target Variable", "Liked")
#         st.metric("ML Task", "Classification")

#     st.divider()

#ml
    st.header("Machine Learning Algorithm")
    st.write("""
Random Forest Classifier

Random Forest is a supervised Machine Learning algorithm used for
classification problems.

It builds multiple decision trees and combines their predictions to
produce accurate and reliable results.

Reasons for selecting Random Forest:

✔ High Accuracy

✔ Fast Prediction

✔ Less Overfitting

✔ Handles Large Dataset

✔ Suitable for Classification Problems
""")
    st.divider()

#technologies
    st.header("Technologies Used")
    st.write("""
    - Python – Programming Language
    - Streamlit – Web Application Framework
    - Pandas – Data Processing
    - Scikit-Learn – Machine Learning
    - Matplotlib – Data Visualization
    - Streamlit Option Menu – Navigation Menu
    """)
    st.divider()

#advantages
    st.header("Advantages")
    st.write("""
• Predicts user movie preferences.

• Easy to use interface.

• Fast Machine Learning prediction.

• Interactive dashboard.

• Useful for personalized movie recommendation.
""")
    st.divider()

#future scope
    st.header("(┬┬﹏┬┬)Future Scope")
    st.write("""
• Deep Learning based recommendation.

• Real-time movie database integration.

• User Login System.

• Movie Posters.

• Online Recommendation Engine.

• Personalized Watchlist.

• OTT Platform Integration.
""")
    st.divider()


#doc_file = "MovieMatrix Documentation.docx"

#with open(doc_file, "rb") as file:
  #  st.download_button(
    #    label="📥 Download Documentation",
     #   data=file,
       # file_name="MovieMatrix Documentation.docx",
        #mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    #)    
#developer
    st.header("🧑‍💻Developer")
    st.success("""
Project Title : MovieMatrix

Develop By Dhrumil Patel & Rishi Suthar

Developed Using Python, Streamlit and Machine Learning

Algorithm : Random Forest Classifier
""")
    st.divider()
