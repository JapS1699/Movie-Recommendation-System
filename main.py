import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Movie data
data = {
    "userId": [1, 1, 1, 2, 2, 3, 3, 4, 5, 5],
    "movieId": [101, 102, 103, 101, 104, 102, 103, 104, 105, 106],
    "rating": [4, 5, 2, 5, 3, 4, 1, 5, 4, 3],
    "genre": [
        "Action", "Comedy", "Action", "Action", "Drama",
        "Comedy", "Action", "Drama", "Science Fiction", "Horror"
    ]
}
df = pd.DataFrame(data)

# List of genres
genres = [
    "Action", "Animation", "Comedy", "Crime", "Drama",
    "Experimental", "Fantasy", "Historical", "Horror",
    "Romance", "Science Fiction", "Thriller", "Western",
    "Musical", "War"
]

# Show genres to the user
print("Welcome to the Movie Recommender!")
print("Here are the genres you can choose from:")
for g in genres:
    print(f"- {g}")

# Ask the user for their preferred genre
while True:
    chosen_genre = input("\nWhat genre of movie would you like to watch? ").strip().capitalize()
    if chosen_genre in genres:
        break
    print("Oops, that's not on the list. Try again!")

# Filter movies based on the genre
filtered_movies = df[df["genre"] == chosen_genre]["movieId"].unique()

if len(filtered_movies) == 0:
    print(f"Sorry, no movies are available in the {chosen_genre} genre right now. :(")
else:
    print(f"\nGreat choice! Here are the movies we found in the {chosen_genre} genre: {list(filtered_movies)}")

    # Prepare data for recommendation
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    # Split data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # Train the recommendation model
    model = SVD()
    print("\nTraining the model... (This might take a second)")
    model.fit(trainset)

    # Evaluate the model
    predictions = model.test(testset)
    print("\nModel trained! Here's how accurate it is:")
    rmse(predictions)

    # Ask the user for their user ID to provide recommendations
    try:
        user_id = int(input("\nEnter your user ID to get recommendations: "))
    except ValueError:
        print("Invalid input. Please enter a number!")
        exit()

    # Find movies the user hasn't rated yet in their chosen genre
    user_rated = df[df["userId"] == user_id]["movieId"].tolist()
    unrated_movies = [movie for movie in filtered_movies if movie not in user_rated]

    if len(unrated_movies) == 0:
        print("\nYou've already rated all the movies in this genre. No recommendations to give!")
    else:
        # Predict ratings for unrated movies
        print("\nHere are the top recommended movies for you:")
        predictions = [(movie, model.predict(user_id, movie).est) for movie in unrated_movies]
        predictions.sort(key=lambda x: x[1], reverse=True)

        for movie, rating in predictions[:3]:  # Top 3 recommendations
            print(f"- Movie {movie}: Predicted rating {rating:.2f}")
