Invoice Expense Classification API

A lightweight, high-performance Python API that classifies invoice text into correct expense categories using Scikit-Learn and FastAPI.

🎯 Project Objective

This API exposes a POST /predict endpoint that takes raw invoice text and predicts its matching expense category, along with a confidence score.

Supported Categories:

Logistics (e.g., courier charges, shipping fees)

Office Supplies (e.g., desk accessories, paper, stationery)

Cloud/Software (e.g., AWS bills, software licenses)

Utilities (e.g., electricity, water, internet charges)

Travel (e.g., flights, hotel bookings, taxi rides)

Inventory (e.g., raw materials, manufacturing components)

💻 Local Setup & Installation

Follow these steps to run the project on your local machine:

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install all required packages:

pip install -r requirements.txt


Train the machine learning model:
This reads data.csv and saves the trained classifier as model.joblib.

python3 train.py


Start the FastAPI application server:

python3 -m uvicorn main:app --reload


Test the API interactively:
Open your browser and navigate to: http://127.0.0.1:8000/docs

🐳 Containerization (Docker)

To build and run this application inside a self-contained environment:

Build the Docker image:

docker build -t invoice-classifier .


Run the Docker container:

docker run -p 8000:8000 invoice-classifier


Once running, you can access the API interactive docs at: http://localhost:8000/docs

📡 API Reference

Predict Expense Category

URL: /predict

Method: POST

Headers: Content-Type: application/json

Request Body Example:

{
  "text": "AWS monthly cloud hosting bill"
}


Response Body Example:

{
  "category": "Cloud/Software",
  "confidence": 0.94
}


📂 Repository Contents

main.py — The primary FastAPI application code containing the API endpoint and prediction logic.

train.py — The model training script that preprocesses data and saves the pipeline.

data.csv — The initial sample dataset used for model training.

requirements.txt — List of required Python packages.

Dockerfile — Docker instruction file to run the service in a container.

README.md — Setup, documentation, and installation guide.
