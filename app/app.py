from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Load CSV files
# -------------------------------------------------------------------
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# -------------------------------------------------------------------
# Database Configuration (MySQL)
# -------------------------------------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------------------------------------------------------
# Database Models
# -------------------------------------------------------------------
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def truncate(text, length=30):
    if len(text) > length:
        return text[:length] + "..."
    return text

# -------------------------------------------------------------------
# Content Based Recommendation System
# -------------------------------------------------------------------
def content_based_recommendations(train_data, item_name, top_n=10):

    matches = train_data[
        train_data['Name'].str.contains(item_name, case=False, na=False)
    ]

    if matches.empty:
        return pd.DataFrame()

    # Use first matching product as seed
    item_index = matches.index[0]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(train_data['Tags'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    similar_items = list(enumerate(cosine_sim[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    top_items = similar_items[1:top_n + 1]
    indices = [i[0] for i in top_items]

    return train_data.iloc[indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ]


# -------------------------------------------------------------------
# Random Images
# -------------------------------------------------------------------
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    random_images = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    return render_template(
        "index.html",
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_images,
        random_price=random.choice(prices)
    )

@app.route("/index")
def index_redirect():
    return index()

@app.route("/main")
def main():
    return render_template(
        "main.html",
        content_based_rec=None,   # 👈 important
        truncate=truncate,
        random_product_image_urls=[],
        random_price=None
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        user = Signup(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()

        return index()

    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["signinUsername"]
        password = request.form["signinPassword"]

        user = Signin(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return index()

    return render_template("signin.html")

@app.route("/recommendations", methods=["POST"])
def recommendations():
    product_name = request.form.get("prod")
    number = int(request.form.get("nbr"))

    recommendations_df = content_based_recommendations(
        train_data, product_name, number
    )

    if recommendations_df.empty:
     return render_template(
        "main.html",
        message="No recommendations found.",
        content_based_rec=None,    
        truncate=truncate,
        random_product_image_urls=[],
        random_price=None
    )


    random_images = [random.choice(random_image_urls) for _ in range(len(recommendations_df))]

    return render_template(
        "main.html",
        content_based_rec=recommendations_df,
        truncate=truncate,
        random_product_image_urls=random_images,
        random_price=random.choice(prices)
    )

# -------------------------------------------------------------------
# Run App
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
