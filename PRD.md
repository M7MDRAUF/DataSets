

Of course. Here is the complete and final Product Requirements Document (PRD) for the CineMatch V1.0.0 demo, written entirely in English.

---

### **Product Requirements Document (PRD): CineMatch V1.0.0**

**Document Version:** 1.1.0
**Date:** October 26, 2023
**Author:** [Your Name/Team Name]
**Status:** Draft

---

### **1. Vision & Introduction**

**Vision:** To build an intelligent and intuitive movie recommendation engine demo, "CineMatch," that showcases the practical application of collaborative filtering. The goal is to move beyond a simple algorithm and present a polished, interactive experience that demonstrates a deep understanding of both data science principles and professional software engineering practices.

**"Why":** Traditional movie platforms overwhelm users with choice. CineMatch V1.0.0 aims to solve this "analysis paralysis" by providing highly relevant, personalized movie suggestions in a clean and engaging interface, proving the power of machine learning in enhancing user discovery.

---

### **2. Problem Statement**

For a user exploring a vast movie catalog, it is difficult to discover new films that align with their unique taste. Existing generic "Top 10" lists are not personalized. Our problem is to leverage the MovieLens dataset to create a system that predicts a user's preference for unseen movies and presents these recommendations in a compelling and explainable manner.

---

### **3. Goals & Success Metrics for V1.0.0 Demo**

| Goal Category | Goal Description | Success Metric |
| :--- | :--- | :--- |
| **Technical** | Implement a robust collaborative filtering model. | Achieve an RMSE of < 0.87 on the test set. |
| **Product/Demo** | Create a seamless and impressive interactive demo. | The professor can navigate the demo without guidance and understands its value within 60 seconds. |
| **User Experience** | Provide clear and useful recommendations. | The "Why are you recommending this?" feature provides a logical explanation for at least 80% of the recommendations. |
| **Professionalism** | Deliver a polished, well-structured, and robust project. | The project runs from a single command (`docker-compose up`), is well-documented, and includes automated checks for data integrity. |

---

### **4. Target Audience**

*   **Primary:** The course professor. He values technical depth, clean code, innovative features, and a clear presentation of the results.
*   **Secondary:** Fellow students and potential future employers. The demo should serve as a strong portfolio piece.

---

### **5. Core Features (MVP for V1.0.0)**

We will use the MoSCoW method for prioritization.

#### **5.1. Must-Haves (Core functionality for the demo)**

| ID | Feature | User Story | Description |
| :--- | :--- | :--- | :--- |
| F-01 | **User Input** | As a user, I want to input my User ID so I can get personalized recommendations. | A simple text input field where the professor can enter any valid User ID from the dataset (e.g., 1-200948). The system should handle invalid IDs gracefully. |
| F-02 | **Recommendation Generation** | As a user, I want to see a list of recommended movies based on my rating history. | The core engine. Upon user ID input, the system queries the pre-trained model to predict ratings for all unseen movies and returns the top N (e.g., N=10). |
| F-03 | **Recommendation Display** | As a user, I want to see the recommended movies with their titles, genres, and predicted ratings. | A clean, card-based layout displaying the movie poster (if possible via API), title, year, genres, and the predicted rating (e.g., "Predicted: 4.5 stars"). |
| F-04 | **Model Pre-computation** | As a developer, I want the model to be pre-trained to ensure instant recommendations in the demo. | The heavy lifting (training the SVD model) is done offline. The demo loads the *trained model object*, not the raw data, to ensure fast response times. |

#### **5.2. Should-Haves (Important features for impact)**

| ID | Feature | User Story | Description |
| :--- | :--- | :--- | :--- |
| F-05 | **Explainable AI (XAI)** | As a user, I want to know *why* a movie is being recommended to me. | This is a key "wow" factor. For each recommendation, show a short explanation. Examples: "Because you highly rated *The Matrix* and *Inception*." or "Users similar to you also loved this film." |
| F-06 | **User Taste Profile** | As a user, I want to see a summary of my own movie taste. | A small dashboard section showing the user's top-rated genres, average rating given, and most highly-rated movies. This provides context for the recommendations. |
| F-07 | **"Surprise Me" Button** | As a user, I want an option to get recommendations that are outside my usual taste. | A button that triggers a different recommendation strategy, perhaps focusing on highly-rated movies from genres the user rarely watches but are similar to movies they *did* like. |
| F-08 | **Feedback Loop** | As a user, I want to give feedback on a recommendation (like/dislike) to see if the system can adapt. | A simple like/dislike button on each recommendation card. Clicking it will not retrain the model (too slow for a demo) but can trigger a message like "Thanks for your feedback! This helps us improve." (Simulating the loop). |

#### **5.3. Could-Haves (Features to show forward-thinking)**

| ID | Feature | User Story | Description |
| :--- | :--- | :--- | :--- |
| F-09 | **Data Visualization** | As a user, I want to see interesting visualizations about the dataset. | A separate "Analytics" page showing charts like: distribution of movie genres, number of ratings per year, a heatmap of user activity. |
| F-10 | **Movie Similarity Explorer** | As a user, I want to select a movie I love and find similar movies. | An input field for a `movieId` or `movieTitle` that returns a list of the most similar movies based on the learned item vectors from the matrix factorization model. |

---

### **6. Technical Requirements & Architecture**

*   **Backend Language:** Python 3.9+
*   **Core Libraries:**
    *   `pandas`: For data manipulation.
    *   `scikit-surprise`: The primary library for building and using the SVD model.
    *   `joblib` or `pickle`: To save and load the pre-trained model object.
*   **Frontend Framework:** **Streamlit**. The perfect choice for a fast, impressive, and Python-based data science demo.
*   **Data Source:** `ml-32m` dataset files (`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`).
*   **Deployment:** **Docker**. Non-negotiable for a professional demo. A `Dockerfile` and `docker-compose.yml` will ensure a one-command setup.

---

### **7. Non-Functional & Development Requirements**

This section specifies requirements not related to direct user functionality, but essential for ensuring the system's robustness and reliability.

| ID | Requirement | Description |
| :--- | :--- | :--- |
| **NF-01** | **Data Integrity & Developer Alerting** | Upon application startup, the system must perform an automated check to verify the presence of all required dataset files (`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`) in the correct path (`data/ml-32m/`).<br><br>**Success Case:** The application proceeds normally, printing a simple informational message to the terminal like `[INFO] All required dataset files found.`<br><br>**Failure Case (if one or more files are missing):**<br>1. The application **fails gracefully** (instead of crashing with a `FileNotFoundError`).<br>2. It displays a **clear and detailed error message** to the developer in both the terminal and the Streamlit UI.<br>3. The message must include:<br>   - The name(s) of the missing file(s).<br>   - The exact path where the file(s) are expected.<br>   - **Actionable instructions:** "Please download the MovieLens ml-32m dataset from the following link: [http://grouplens.org/datasets/movielens/latest/](http://grouplens.org/datasets/movielens/latest/) and place the extracted files in the `data/ml-32m/` directory."<br>4. The program must halt execution after displaying this message to prevent subsequent errors. |

---

### **8. Project Structure (Professional Level)**

A clean project structure is crucial.

```
/cinematch-demo/
|
├── data/                     # All data files
│   ├── ml-32m/               # <-- The system will check this folder
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   └── ...
│   └── processed/            # For any processed data (e.g., user_genre_matrix.pkl)
│
├── models/                   # To store trained models
│   └── svd_model.pkl
│
├── src/                      # All source code
│   ├── __init__.py
│   ├── data_processing.py    # <-- The data integrity check (NF-01) should be implemented here.
│   ├── model_training.py     # Script to train and save the model (run once)
│   ├── recommendation_engine.py # Core logic to get recommendations
│   └── utils.py              # Helper functions (e.g., for explainability)
│
├── app/                      # Streamlit application
│   ├── 1_🏠_Home.py
│   ├── 2_🎬_Recommend.py
│   ├── 3_📊_Analytics.py
│   └── main.py               # Main Streamlit file that runs the app
│
├── scripts/                  # Utility scripts
│   └── train_model.sh        # A script to run the training process
│
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md                 # **CRITICAL**: How to set up and run the project, including a note about the data check.
└── .gitignore
```

---

### **9. V1.0.0 Demo Flow (The Script for the Professor)**

1.  **Introduction:** "Professor, today we present CineMatch, a movie recommendation engine. It's built using collaborative filtering on the MovieLens 32M dataset. We've containerized it for easy deployment and have also included built-in checks to ensure all dependencies, including the dataset, are correctly configured."
2.  **Run:** "You can run the entire application on your machine with this single command: `docker-compose up`." (Show the command).
3.  **Live Demo:**
    *   Open the local URL provided by Streamlit.
    *   **Page 1 (Home):** Briefly explain the project's goal and show a key visualization (e.g., top genres).
    *   **Page 2 (Recommend):**
        *   "Let's pick a random user, say User ID `123`."
        *   Enter `123` and click "Get Recommendations".
        *   "Here are the top 10 recommendations for this user. You can see the predicted rating and the genres."
        *   **(The Wow Moment):** "But *why* was 'The Shawshank Redemption' recommended? Let's click the 'Explain' button." (Show the explanation: "Because you highly rated 'Pulp Fiction' and 'The Godfather'").
        *   "We can also see this user's taste profile here on the side. They clearly love Drama and Crime."
    *   **Page 3 (Analytics):** "Finally, we built a small analytics page to showcase insights from the 32 million ratings we analyzed." (Show a chart).

---

### **10. Out of Scope for V1.0.0**

*   User authentication and profiles.
*   Real-time model retraining based on user feedback.
*   Integrating live APIs for movie posters (this is a "could-have" if time permits, but requires API keys).
*   Handling a continuous stream of new data.

---

### **Final Word (For the Professor)**

"This project demonstrates our ability to bridge the gap between a theoretical algorithm and a functional, user-centric product. We've focused not just on model accuracy (RMSE), but also on user experience, explainability, and professional software engineering practices like containerization, structured development, and robust error handling. We believe CineMatch is a strong testament to our skills as data scientists and engineers."