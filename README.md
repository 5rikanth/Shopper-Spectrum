# Shopper Spectrum: Customer Segmentation and Product Recommendation System

## Project Overview

Shopper Spectrum is an end-to-end Machine Learning project that analyzes e-commerce transaction data to identify customer purchasing patterns, segment customers using RFM Analysis and K-Means Clustering, and generate personalized product recommendations using Item-Based Collaborative Filtering.

The project combines customer analytics, unsupervised machine learning, recommendation systems, and deployment through a Streamlit web application to deliver actionable business insights for e-commerce businesses.

---

## Problem Statement

E-commerce businesses generate vast amounts of transaction data daily. Analyzing this data can help organizations understand customer behavior, improve customer retention, increase sales, and provide personalized shopping experiences.

The objective of this project is to:

- Segment customers based on purchasing behavior using RFM Analysis.
- Identify valuable, regular, occasional, and at-risk customers.
- Build a product recommendation system using collaborative filtering.
- Deploy an interactive web application for real-time predictions and recommendations.

---

## Dataset Information

The project uses an Online Retail Transaction Dataset containing customer purchase records.

### Features

| Feature | Description |
|----------|-------------|
| InvoiceNo | Transaction Number |
| StockCode | Product Identifier |
| Description | Product Name |
| Quantity | Quantity Purchased |
| InvoiceDate | Transaction Date and Time |
| UnitPrice | Product Price |
| CustomerID | Unique Customer Identifier |
| Country | Customer Location |

### Dataset Statistics

- 541,909 total transactions
- 38 countries
- 4,338 customers after preprocessing
- 3,896 products
- 397,884 cleaned transaction records

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed records with missing Customer IDs.
- Removed cancelled transactions.
- Removed records with non-positive quantities and prices.
- Converted InvoiceDate into datetime format.
- Created TotalAmount feature.
- Extracted time-based features such as month, hour, and day of week.
- Generated Recency, Frequency, and Monetary (RFM) metrics.

---

## Exploratory Data Analysis

Comprehensive EDA was performed to understand customer behavior and sales trends.

Key analyses include:

- Transaction value distribution
- Country-wise transaction volume
- Revenue by country
- Monthly sales trends
- Day-wise and hourly purchasing behavior
- Top-selling products
- RFM distributions
- Correlation analysis
- Customer segment profiling

More than 15 visualizations were created to derive business insights.

---

## Feature Engineering

### RFM Analysis

Customer behavior was represented using three metrics:

- Recency: Days since the customer's last purchase
- Frequency: Number of purchases made by the customer
- Monetary: Total amount spent by the customer

These features were standardized using StandardScaler before clustering.

---

## Machine Learning Approach

### Clustering Algorithms Evaluated

- K-Means Clustering
- Agglomerative Hierarchical Clustering
- DBSCAN

### Model Selection

K-Means with 4 clusters was selected as the final model based on:

- Higher Silhouette Score
- Better cluster separation
- Improved business interpretability

### Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Score
- Elbow Method

---

## Customer Segments

The clustering process identified four meaningful customer groups:

| Segment | Description |
|----------|-------------|
| High-Value | Recent, frequent, high-spending customers |
| Regular | Consistent customers with moderate spending |
| Occasional | Infrequent customers with lower spending |
| At-Risk | Customers likely to churn |

---

## Recommendation System

An Item-Based Collaborative Filtering approach was implemented to recommend products.

### Workflow

1. Create Customer–Product Interaction Matrix
2. Calculate Cosine Similarity between products
3. Identify similar products
4. Return Top 5 recommendations

The recommendation engine provides personalized product suggestions based on historical purchasing patterns.

---

## Streamlit Application

The project is deployed as an interactive Streamlit application.

### Product Recommendation Module

Input:
- Product Name

Output:
- Top 5 Similar Products
- Similarity Scores

### Customer Segmentation Module

Input:
- Recency
- Frequency
- Monetary

Output:
- Predicted Customer Segment
- Segment Description
- Business Recommendations

### Dashboard Features

- Customer Metrics
- Product Catalog Statistics
- Segment Distribution
- Real-Time Predictions

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- SciPy
- Joblib
- Streamlit

### Machine Learning Techniques

- RFM Analysis
- K-Means Clustering
- Agglomerative Clustering
- DBSCAN
- PCA
- Collaborative Filtering
- Cosine Similarity

---

## Project Structure

```text
Shopper-Spectrum/
│
├── app.py
├── Shopper_Spectrum_Notebook.ipynb
├── rfm_data.csv
├── requirements.txt
│
└── models/
    ├── kmeans_model.pkl
    ├── scaler.pkl
    ├── item_similarity.pkl
    └── rfm_data.csv
```

---

## Installation and Setup

### Clone Repository

```bash
git clone https://github.com/5rikanth/Shopper-Spectrum.git
cd Shopper-Spectrum
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Key Business Insights

- The United Kingdom contributes the majority of transactions and revenue.
- Customer spending is highly concentrated among a small group of high-value customers.
- Most customers belong to the Occasional segment.
- Purchase activity peaks during the holiday season.
- Frequent customers tend to generate significantly higher revenue.
- Collaborative filtering can be used to improve cross-selling opportunities.

---

## Results

- Successfully segmented customers into four actionable business groups.
- Achieved strong clustering performance using K-Means.
- Built a recommendation engine using item-based collaborative filtering.
- Developed a deployment-ready Streamlit application for real-time recommendations and segmentation.

---

## Skills Demonstrated

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Customer Analytics
- RFM Analysis
- Unsupervised Machine Learning
- Clustering Evaluation
- Recommendation Systems
- Data Visualization
- Streamlit Deployment
- Business Intelligence

---
