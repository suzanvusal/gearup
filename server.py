# server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "gearup_repairs")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Initialize FastAPI app
app = FastAPI(title="GearUp Repairs API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# ------------------------
# Utility functions
# ------------------------
def get_collection(name: str):
    return db[name]

# ------------------------
# Health Check
# ------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "GearUp Repairs backend is running"}

# ------------------------
# Users / Consumers
# ------------------------
@app.get("/users")
def list_users():
    users = list(get_collection("users").find({}, {"_id": 0}))
    return {"users": users}

@app.get("/users/{email}")
def get_user(email: str):
    user = get_collection("users").find_one({"email": email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ------------------------
# Technicians
# ------------------------
@app.get("/technicians")
def list_technicians():
    techs = list(get_collection("technicians").find({}, {"_id": 0}))
    return {"technicians": techs}

# ------------------------
# Parts / Marketplace
# ------------------------
@app.get("/parts")
def list_parts():
    parts = list(get_collection("parts").find({}, {"_id": 0}))
    return {"parts": parts}

# ------------------------
# Bookings
# ------------------------
@app.get("/bookings")
def list_bookings():
    bookings = list(get_collection("bookings").find({}, {"_id": 0}))
    return {"bookings": bookings}

@app.post("/bookings")
def create_booking(booking: dict):
    result = get_collection("bookings").insert_one(booking)
    if not result.acknowledged:
        raise HTTPException(status_code=500, detail="Failed to create booking")
    return {"status": "success", "booking": booking}

# ------------------------
# Reviews
# ------------------------
@app.get("/reviews")
def list_reviews():
    reviews = list(get_collection("reviews").find({}, {"_id": 0}))
    return {"reviews": reviews}

@app.post("/reviews")
def create_review(review: dict):
    result = get_collection("reviews").insert_one(review)
    if not result.acknowledged:
        raise HTTPException(status_code=500, detail="Failed to create review")
    return {"status": "success", "review": review}

# ------------------------
# Example Admin Endpoint
# ------------------------
@app.get("/admin/stats")
def admin_stats():
    return {
        "total_users": get_collection("users").count_documents({}),
        "total_technicians": get_collection("technicians").count_documents({}),
        "total_bookings": get_collection("bookings").count_documents({}),
        "total_parts": get_collection("parts").count_documents({}),
        "total_reviews": get_collection("reviews").count_documents({}),
    }

# ------------------------
# Root Endpoint
# ------------------------
@app.get("/")
def root():
    return {"message": "Welcome to GearUp Repairs API!"}

