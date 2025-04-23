import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
from sqlalchemy import create_engine




# --------------------------
# Veritabanı Bağlantısı (Northwind)
# --------------------------
from dotenv import load_dotenv
import os

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)
# --------------------------
# SQL'den Verileri Çekme
# --------------------------

def fetch_product_data():
    query = """
        SELECT 
            p."ProductID",
            AVG(od."UnitPrice") AS avg_price,
            COUNT(DISTINCT od."OrderID") AS order_freq,
            AVG(od."Quantity") AS avg_quantity_per_order,
            COUNT(DISTINCT o."CustomerID") AS distinct_customers
        FROM "Products" p
        JOIN "Order Details" od ON p."ProductID" = od."ProductID"
        JOIN "Orders" o ON od."OrderID" = o."OrderID"
        GROUP BY p."ProductID"
    """
    return pd.read_sql(query, engine)

def fetch_supplier_data():
    query = """
        SELECT 
            s."SupplierID",
            COUNT(DISTINCT p."ProductID") AS product_count,
            SUM(od."Quantity") AS total_sales_volume,
            AVG(od."UnitPrice") AS avg_price,
            COUNT(DISTINCT o."CustomerID") AS avg_customers
        FROM "Suppliers" s
        JOIN "Products" p ON s."SupplierID" = p."SupplierID"
        JOIN "Order Details" od ON p."ProductID" = od."ProductID"
        JOIN "Orders" o ON od."OrderID" = o."OrderID"
        GROUP BY s."SupplierID"
    """
    return pd.read_sql(query, engine)

def fetch_country_data():
    query = """
        SELECT 
            c."Country",
            COUNT(o."OrderID") AS total_orders,
            AVG(od."UnitPrice" * od."Quantity") AS avg_order_value,
            AVG(sub.count_items) AS avg_items_per_order
        FROM "Customers" c
        JOIN "Orders" o ON c."CustomerID" = o."CustomerID"
        JOIN "Order Details" od ON o."OrderID" = od."OrderID"
        JOIN (
            SELECT "OrderID", COUNT(*) AS count_items
            FROM "Order Details"
            GROUP BY "OrderID"
        ) sub ON sub."OrderID" = o."OrderID"
        GROUP BY c."Country"
    """
    return pd.read_sql(query, engine)

# --------------------------
# DBSCAN Fonksiyonu
# --------------------------

def apply_dbscan(df, feature_columns, eps=0.5, min_samples=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_columns])
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = model.fit_predict(X_scaled)
    return df

# --------------------------
# min_samples için AR-GE
# --------------------------

def optimize_min_samples(df, features, eps=0.5, min_samples_range=range(2, 10)):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    results = []
    for min_samples in min_samples_range:
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = model.labels_
        non_noise = labels != -1
        if np.unique(labels[non_noise]).size > 1:
            score = silhouette_score(X_scaled[non_noise], labels[non_noise])
        else:
            score = -1
        results.append((min_samples, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# --------------------------
# FastAPI Uygulaması
# --------------------------

app = FastAPI()

class ClusterRequest(BaseModel):
    target: Literal["product", "supplier", "country"]
    eps: float = 0.5
    min_samples: int = 3

@app.post("/cluster/")
def cluster(request: ClusterRequest):
    if request.target == "product":
        df = fetch_product_data()
        features = ["avg_price", "order_freq", "avg_quantity_per_order", "distinct_customers"]
    elif request.target == "supplier":
        df = fetch_supplier_data()
        features = ["product_count", "total_sales_volume", "avg_price", "avg_customers"]
    elif request.target == "country":
        df = fetch_country_data()
        features = ["total_orders", "avg_order_value", "avg_items_per_order"]
    else:
        return {"error": "Invalid target"}

    clustered = apply_dbscan(df, features, request.eps, request.min_samples)
    return clustered.to_dict(orient="records")

@app.post("/optimize/")
def optimize(request: ClusterRequest):
    if request.target == "product":
        df = fetch_product_data()
        features = ["avg_price", "order_freq", "avg_quantity_per_order", "distinct_customers"]
    elif request.target == "supplier":
        df = fetch_supplier_data()
        features = ["product_count", "total_sales_volume", "avg_price", "avg_customers"]
    elif request.target == "country":
        df = fetch_country_data()
        features = ["total_orders", "avg_order_value", "avg_items_per_order"]
    else:
        return {"error": "Invalid target"}

    optimization = optimize_min_samples(df, features, request.eps)
    return {"results": optimization}


