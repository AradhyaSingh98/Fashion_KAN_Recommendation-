# Fashion Recommendation System using KAN

## About the Project

This project is a Fashion Recommendation System developed using the H&M Fashion Dataset. The main goal is to recommend fashion products based on user preferences like age, product category, colour, department, and price.

Along with recommendation using the KAN model, the project also provides an image-based recommendation feature using ResNet50. A simple Streamlit interface is used so users can easily interact with the system.

## Features

- Recommend products based on user details
- Image-based fashion recommendation
- Compare KAN with MLP
- Simple and interactive Streamlit interface
- Product filtering based on price

## Technologies Used

- Python
- PyTorch
- KAN (Kolmogorov-Arnold Network)
- ResNet50
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pillow

## Dataset

The project uses the H&M Fashion Dataset which contains:

- Customer details
- Product information
- Transaction history
- Fashion product images

## How the Project Works

1. The dataset is cleaned and preprocessed.
2. Important features are selected.
3. The KAN model is trained for product recommendation.
4. ResNet50 is used to extract image features.
5. Similar images are found using cosine similarity.
6. Everything is connected through a Streamlit web application.

## Project Modules

### KAN Recommendation

Users enter their age and maximum budget. The trained KAN model predicts recommendation scores and displays the top matching products.

### Image Recommendation

Users upload a fashion image. The system extracts image features using ResNet50 and finds visually similar products from the dataset.

### KAN vs MLP

This section compares the performance of the KAN model with a traditional MLP model using different evaluation metrics.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

## Results

The KAN model performed competitively and produced good recommendations. It also showed better overall performance than the baseline MLP model in most evaluation metrics.

## How to Run

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Future Improvements

- Better personalized recommendations
- Real-time recommendation updates
- User login system
- Deployment on cloud
- Mobile application support

## Author

**Aradhya Singh**

B.Tech Student

Machine Learning & AI Enthusiast