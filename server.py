from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Header
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timezone, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionResponse, CheckoutStatusResponse, CheckoutSessionRequest
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

EMERGENT_LLM_KEY = os.environ['EMERGENT_LLM_KEY']
STRIPE_API_KEY = os.environ['STRIPE_API_KEY']
PLATFORM_BOOKING_COMMISSION = float(os.environ.get('PLATFORM_BOOKING_COMMISSION', '0.15'))
PLATFORM_PARTS_COMMISSION = float(os.environ.get('PLATFORM_PARTS_COMMISSION', '0.10'))

# ============ MODELS ============

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    picture: Optional[str] = None
    role: str  # consumer, technician, admin
    location: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Consumer fields
    equipment: Optional[List[Dict]] = []  # [{name, model, brand, condition, purchase_date}]
    # Technician fields
    services_offered: Optional[List[str]] = []  # [repair, installation, inspection, etc.]
    qualifications: Optional[str] = None
    certifications: Optional[List[str]] = []
    availability: Optional[str] = None
    hourly_rate: Optional[float] = None
    rating: Optional[float] = 0.0
    total_reviews: Optional[int] = 0

class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Booking(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    consumer_id: str
    technician_id: Optional[str] = None
    service_type: str  # repair, installation, inspection, etc.
    equipment_details: Dict  # {name, model, issue_description}
    preferred_date: Optional[str] = None
    status: str = "pending"  # pending, accepted, in_progress, completed, cancelled, paid
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    payment_required: bool = False
    payment_status: str = "unpaid"  # unpaid, paid
    payment_session_id: Optional[str] = None
    location: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

class Part(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str  # motor, belt, cable, electronics, etc.
    brand: str
    compatible_models: List[str]
    description: str
    price: float
    supplier_id: str
    supplier_name: str
    stock_quantity: int
    image_url: Optional[str] = None
    rating: Optional[float] = 0.0
    total_reviews: Optional[int] = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str
    reviewer_name: str
    target_type: str  # technician, part, supplier
    target_id: str
    rating: float  # 1-5
    comment: Optional[str] = None
    booking_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PaymentTransaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    amount: float
    currency: str
    payment_status: str  # initiated, paid, failed, expired
    status: str  # initiated, completed, failed
    metadata: Optional[Dict] = {}
    booking_id: Optional[str] = None
    part_ids: Optional[List[str]] = []
    technician_id: Optional[str] = None
    platform_commission: Optional[float] = 0.0
    technician_payout: Optional[float] = 0.0
    supplier_payout: Optional[float] = 0.0
    transaction_type: Optional[str] = "parts"  # parts, service
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

# ============ AUTH ENDPOINTS ============

@api_router.get("/auth/session")
async def get_session(
    request: Request,
    x_session_id: Optional[str] = Header(None)
):
    """Process session_id from Emergent Auth"""
    try:
        if not x_session_id:
            raise HTTPException(401, "No session ID provided")
        
        # Call Emergent auth API
        auth_api_url = os.environ.get('AUTH_API_URL', 'https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data')
        async with httpx.AsyncClient() as client:
            response = await client.get(
                auth_api_url,
                headers={"X-Session-ID": x_session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(401, "Invalid session")
            
            user_data = response.json()
        
        # Check if user exists
        existing_user = await db.users.find_one({"email": user_data["email"]}, {"_id": 0})
        
        if not existing_user:
            # Create new user
            new_user = User(
                email=user_data["email"],
                name=user_data["name"],
                picture=user_data.get("picture"),
                role="consumer"  # default role
            )
            user_dict = new_user.model_dump()
            user_dict['created_at'] = user_dict['created_at'].isoformat()
            await db.users.insert_one(user_dict)
            user_id = new_user.id
        else:
            user_id = existing_user["id"]
        
        # Create session
        session_token = user_data["session_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        new_session = Session(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at
        )
        
        session_dict = new_session.model_dump()
        session_dict['created_at'] = session_dict['created_at'].isoformat()
        session_dict['expires_at'] = session_dict['expires_at'].isoformat()
        
        await db.sessions.insert_one(session_dict)
        
        return {
            "user_id": user_id,
            "email": user_data["email"],
            "name": user_data["name"],
            "picture": user_data.get("picture"),
            "session_token": session_token
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Auth error: {e}")
        raise HTTPException(500, str(e))

async def get_current_user(request: Request) -> User:
    """Helper to get current user from session"""
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    
    if not session_token:
        raise HTTPException(401, "Not authenticated")
    
    session = await db.sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session:
        raise HTTPException(401, "Invalid session")
    
    if datetime.fromisoformat(session["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(401, "Session expired")
    
    user = await db.users.find_one({"id": session["user_id"]}, {"_id": 0})
    
    if not user:
        raise HTTPException(404, "User not found")
    
    if isinstance(user.get('created_at'), str):
        user['created_at'] = datetime.fromisoformat(user['created_at'])
    
    return User(**user)

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user"""
    session_token = request.cookies.get("session_token")
    
    if session_token:
        await db.sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie("session_token")
    return {"message": "Logged out"}

# ============ USER ENDPOINTS ============

@api_router.get("/users/me", response_model=User)
async def get_current_user_profile(request: Request):
    """Get current user profile"""
    return await get_current_user(request)

@api_router.put("/users/me")
async def update_user_profile(request: Request, updates: Dict):
    """Update user profile"""
    user = await get_current_user(request)
    
    # Update user
    await db.users.update_one(
        {"id": user.id},
        {"$set": updates}
    )
    
    return {"message": "Profile updated"}

@api_router.get("/technicians", response_model=List[User])
async def get_technicians(location: Optional[str] = None):
    """Get list of technicians"""
    query = {"role": "technician"}
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    
    technicians = await db.users.find(query, {"_id": 0}).to_list(100)
    
    for tech in technicians:
        if isinstance(tech.get('created_at'), str):
            tech['created_at'] = datetime.fromisoformat(tech['created_at'])
    
    return technicians

@api_router.get("/technicians/{technician_id}", response_model=User)
async def get_technician(technician_id: str):
    """Get technician details"""
    tech = await db.users.find_one({"id": technician_id, "role": "technician"}, {"_id": 0})
    
    if not tech:
        raise HTTPException(404, "Technician not found")
    
    if isinstance(tech.get('created_at'), str):
        tech['created_at'] = datetime.fromisoformat(tech['created_at'])
    
    return tech

# ============ BOOKING ENDPOINTS ============

class BookingCreate(BaseModel):
    service_type: str
    equipment_details: Dict
    preferred_date: Optional[str] = None
    location: str
    technician_id: Optional[str] = None

@api_router.post("/bookings", response_model=Booking)
async def create_booking(request: Request, booking_data: BookingCreate):
    """Create service booking"""
    user = await get_current_user(request)
    
    # Use AI to estimate cost if no technician selected
    estimated_cost = None
    if not booking_data.technician_id:
        # AI price prediction
        try:
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"price_estimate_{user.id}",
                system_message="You are a fitness equipment service pricing expert. Provide price estimates in USD."
            ).with_model("openai", "gpt-5")
            
            message = UserMessage(
                text=f"Estimate the repair cost for: {booking_data.service_type} - {booking_data.equipment_details.get('name', 'equipment')} - Issue: {booking_data.equipment_details.get('issue_description', 'general service')}. Reply with just a number range like 50-100."
            )
            
            response = await chat.send_message(message)
            estimated_cost = response
        except Exception as e:
            logging.error(f"AI pricing error: {e}")
            estimated_cost = "Contact technician for quote"
    
    booking = Booking(
        consumer_id=user.id,
        technician_id=booking_data.technician_id,
        service_type=booking_data.service_type,
        equipment_details=booking_data.equipment_details,
        preferred_date=booking_data.preferred_date,
        location=booking_data.location,
        estimated_cost=estimated_cost
    )
    
    booking_dict = booking.model_dump()
    booking_dict['created_at'] = booking_dict['created_at'].isoformat()
    
    await db.bookings.insert_one(booking_dict)
    
    return booking

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(request: Request):
    """Get user bookings"""
    user = await get_current_user(request)
    
    query = {}
    if user.role == "consumer":
        query["consumer_id"] = user.id
    elif user.role == "technician":
        query["technician_id"] = user.id
    
    bookings = await db.bookings.find(query, {"_id": 0}).to_list(100)
    
    for booking in bookings:
        if isinstance(booking.get('created_at'), str):
            booking['created_at'] = datetime.fromisoformat(booking['created_at'])
        if booking.get('completed_at') and isinstance(booking.get('completed_at'), str):
            booking['completed_at'] = datetime.fromisoformat(booking['completed_at'])
    
    return bookings

@api_router.put("/bookings/{booking_id}")
async def update_booking(request: Request, booking_id: str, updates: Dict):
    """Update booking status"""
    user = await get_current_user(request)
    
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(404, "Booking not found")
    
    # Check permissions
    if user.role == "technician" and booking.get("technician_id") != user.id:
        raise HTTPException(403, "Not authorized")
    
    if user.role == "consumer" and booking.get("consumer_id") != user.id:
        raise HTTPException(403, "Not authorized")
    
    # If marking as completed, require payment
    if updates.get("status") == "completed" and user.role == "technician":
        updates["payment_required"] = True
    
    await db.bookings.update_one(
        {"id": booking_id},
        {"$set": updates}
    )
    
    return {"message": "Booking updated"}

@api_router.post("/bookings/{booking_id}/request-payment")
async def request_booking_payment(request: Request, booking_id: str, cost: float):
    """Technician requests payment after completing service"""
    user = await get_current_user(request)
    
    if user.role != "technician":
        raise HTTPException(403, "Only technicians can request payment")
    
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    
    if not booking:
        raise HTTPException(404, "Booking not found")
    
    if booking.get("technician_id") != user.id:
        raise HTTPException(403, "Not authorized")
    
    if booking.get("status") != "completed":
        raise HTTPException(400, "Booking must be completed first")
    
    # Update booking with actual cost and payment required flag
    await db.bookings.update_one(
        {"id": booking_id},
        {"$set": {
            "actual_cost": cost,
            "payment_required": True,
            "payment_status": "unpaid"
        }}
    )
    
    return {"message": "Payment request sent to consumer"}

# ============ PARTS MARKETPLACE ============

@api_router.get("/parts", response_model=List[Part])
async def get_parts(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    search: Optional[str] = None
):
    """Get parts catalog"""
    query = {}
    
    if category:
        query["category"] = category
    if brand:
        query["brand"] = {"$regex": brand, "$options": "i"}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}}
        ]
    
    parts = await db.parts.find(query, {"_id": 0}).to_list(100)
    
    for part in parts:
        if isinstance(part.get('created_at'), str):
            part['created_at'] = datetime.fromisoformat(part['created_at'])
    
    return parts

@api_router.post("/parts/ai-recommend")
async def ai_part_recommendation(request: Request, equipment_info: Dict):
    """AI-powered part recommendations"""
    user = await get_current_user(request)
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"part_rec_{uuid.uuid4()}",
            system_message="You are a fitness equipment parts expert. Recommend compatible parts based on equipment details."
        ).with_model("openai", "gpt-5")
        
        message = UserMessage(
            text=f"Based on this equipment: {equipment_info.get('name')} - {equipment_info.get('model')} - {equipment_info.get('brand')} with issue: {equipment_info.get('issue', 'maintenance')}, what parts might be needed? List 3-5 part types."
        )
        
        response = await chat.send_message(message)
        
        return {"recommendations": response}
    except Exception as e:
        logging.error(f"AI recommendation error: {e}")
        raise HTTPException(500, "AI service unavailable")

# ============ REVIEWS ============

class ReviewCreate(BaseModel):
    target_type: str
    target_id: str
    rating: float
    comment: Optional[str] = None
    booking_id: Optional[str] = None

@api_router.post("/reviews", response_model=Review)
async def create_review(request: Request, review_data: ReviewCreate):
    """Create review"""
    user = await get_current_user(request)
    
    review = Review(
        reviewer_id=user.id,
        reviewer_name=user.name,
        target_type=review_data.target_type,
        target_id=review_data.target_id,
        rating=review_data.rating,
        comment=review_data.comment,
        booking_id=review_data.booking_id
    )
    
    review_dict = review.model_dump()
    review_dict['created_at'] = review_dict['created_at'].isoformat()
    
    await db.reviews.insert_one(review_dict)
    
    # Update target rating
    if review_data.target_type == "technician":
        reviews = await db.reviews.find(
            {"target_id": review_data.target_id, "target_type": "technician"},
            {"_id": 0}
        ).to_list(1000)
        
        avg_rating = sum(r["rating"] for r in reviews) / len(reviews)
        
        await db.users.update_one(
            {"id": review_data.target_id},
            {"$set": {"rating": avg_rating, "total_reviews": len(reviews)}}
        )
    
    return review

@api_router.get("/reviews/{target_id}", response_model=List[Review])
async def get_reviews(target_id: str, target_type: str):
    """Get reviews for target"""
    reviews = await db.reviews.find(
        {"target_id": target_id, "target_type": target_type},
        {"_id": 0}
    ).to_list(100)
    
    for review in reviews:
        if isinstance(review.get('created_at'), str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    
    return reviews

# ============ AI FEATURES ============

@api_router.post("/ai/match-technician")
async def match_technician(request: Request, service_details: Dict):
    """AI-powered technician matching"""
    user = await get_current_user(request)
    
    try:
        # Get available technicians
        location = service_details.get("location", user.location)
        technicians = await db.users.find(
            {"role": "technician", "location": {"$regex": location, "$options": "i"}},
            {"_id": 0}
        ).to_list(50)
        
        if not technicians:
            return {"matches": [], "message": "No technicians found in your area"}
        
        # Use AI to rank technicians
        tech_list = "\n".join([
            f"{t['name']}: Services: {', '.join(t.get('services_offered', []))}, Rating: {t.get('rating', 0)}, Rate: ${t.get('hourly_rate', 0)}/hr"
            for t in technicians[:10]
        ])
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"match_{user.id}",
            system_message="You are a smart matching assistant. Rank technicians based on service type, ratings, and rates."
        ).with_model("openai", "gpt-5")
        
        message = UserMessage(
            text=f"Service needed: {service_details.get('service_type')} for {service_details.get('equipment')}. Available technicians:\n{tech_list}\n\nRank top 3 technicians and explain why."
        )
        
        response = await chat.send_message(message)
        
        return {
            "technicians": technicians[:5],
            "ai_recommendation": response
        }
    except Exception as e:
        logging.error(f"AI matching error: {e}")
        return {"technicians": technicians[:5], "ai_recommendation": "AI service temporarily unavailable"}

@api_router.post("/ai/predictive-maintenance")
async def predictive_maintenance(request: Request, equipment_data: Dict):
    """AI-powered predictive maintenance"""
    user = await get_current_user(request)
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"predict_{uuid.uuid4()}",
            system_message="You are a predictive maintenance expert for fitness equipment."
        ).with_model("openai", "gpt-5")
        
        message = UserMessage(
            text=f"Equipment: {equipment_data.get('name')} - {equipment_data.get('model')}, Purchase date: {equipment_data.get('purchase_date')}, Usage: {equipment_data.get('usage_frequency', 'regular')}. What maintenance should be done and when?"
        )
        
        response = await chat.send_message(message)
        
        return {"maintenance_plan": response}
    except Exception as e:
        logging.error(f"Predictive maintenance error: {e}")
        raise HTTPException(500, "AI service unavailable")

# ============ PAYMENT ENDPOINTS ============

class CheckoutRequest(BaseModel):
    booking_id: Optional[str] = None
    part_ids: Optional[List[str]] = []
    amount: float
    currency: str = "usd"
    origin_url: str

@api_router.post("/payments/checkout")
async def create_checkout(request: Request, checkout_req: CheckoutRequest):
    """Create Stripe checkout session"""
    user = await get_current_user(request)
    
    try:
        # Initialize Stripe
        webhook_url = f"{checkout_req.origin_url}/api/webhook/stripe"
        stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url=webhook_url)
        
        # Calculate commission
        platform_commission = 0.0
        transaction_type = "parts"
        technician_id = None
        
        if checkout_req.booking_id:
            # Service payment
            transaction_type = "service"
            booking = await db.bookings.find_one({"id": checkout_req.booking_id}, {"_id": 0})
            if booking:
                technician_id = booking.get("technician_id")
                platform_commission = checkout_req.amount * PLATFORM_BOOKING_COMMISSION
        else:
            # Parts payment
            platform_commission = checkout_req.amount * PLATFORM_PARTS_COMMISSION
        
        technician_payout = checkout_req.amount - platform_commission if technician_id else 0.0
        supplier_payout = checkout_req.amount - platform_commission if not technician_id else 0.0
        
        # Create checkout session
        success_url = f"{checkout_req.origin_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{checkout_req.origin_url}/payment/cancel"
        
        metadata = {
            "user_id": user.id,
            "booking_id": checkout_req.booking_id or "",
            "part_ids": ",".join(checkout_req.part_ids),
            "transaction_type": transaction_type,
            "technician_id": technician_id or "",
            "platform_commission": str(platform_commission)
        }
        
        checkout_request = CheckoutSessionRequest(
            amount=checkout_req.amount,
            currency=checkout_req.currency,
            success_url=success_url,
            cancel_url=cancel_url,
            metadata=metadata
        )
        
        session = await stripe_checkout.create_checkout_session(checkout_request)
        
        # Create payment transaction
        transaction = PaymentTransaction(
            user_id=user.id,
            session_id=session.session_id,
            amount=checkout_req.amount,
            currency=checkout_req.currency,
            payment_status="initiated",
            status="initiated",
            metadata=metadata,
            booking_id=checkout_req.booking_id,
            part_ids=checkout_req.part_ids,
            technician_id=technician_id,
            platform_commission=platform_commission,
            technician_payout=technician_payout,
            supplier_payout=supplier_payout,
            transaction_type=transaction_type
        )
        
        trans_dict = transaction.model_dump()
        trans_dict['created_at'] = trans_dict['created_at'].isoformat()
        
        await db.payment_transactions.insert_one(trans_dict)
        
        # Update booking if service payment
        if checkout_req.booking_id:
            await db.bookings.update_one(
                {"id": checkout_req.booking_id},
                {"$set": {"payment_required": True, "payment_session_id": session.session_id}}
            )
        
        return {"url": session.url, "session_id": session.session_id}
    
    except Exception as e:
        logging.error(f"Checkout error: {e}")
        raise HTTPException(500, str(e))

@api_router.get("/payments/status/{session_id}")
async def get_payment_status(request: Request, session_id: str):
    """Get payment status"""
    user = await get_current_user(request)
    
    try:
        stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url="")
        status = await stripe_checkout.get_checkout_status(session_id)
        
        # Update transaction
        transaction = await db.payment_transactions.find_one(
            {"session_id": session_id, "user_id": user.id},
            {"_id": 0}
        )
        
        if transaction and transaction.get("payment_status") != "paid":
            update_data = {
                "payment_status": status.payment_status,
                "status": "completed" if status.payment_status == "paid" else "failed",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            await db.payment_transactions.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            # Update booking if service payment and paid
            if status.payment_status == "paid" and transaction.get("booking_id"):
                await db.bookings.update_one(
                    {"id": transaction["booking_id"]},
                    {"$set": {"payment_status": "paid", "status": "paid"}}
                )
        
        return {
            "status": status.status,
            "payment_status": status.payment_status,
            "amount": status.amount_total / 100,
            "currency": status.currency
        }
    
    except Exception as e:
        logging.error(f"Payment status error: {e}")
        raise HTTPException(500, str(e))

@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    try:
        body = await request.body()
        signature = request.headers.get("Stripe-Signature")
        
        stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url="")
        webhook_response = await stripe_checkout.handle_webhook(body, signature)
        
        # Update transaction based on webhook
        await db.payment_transactions.update_one(
            {"session_id": webhook_response.session_id},
            {"$set": {
                "payment_status": webhook_response.payment_status,
                "status": "completed" if webhook_response.payment_status == "paid" else "failed",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return {"received": True}
    
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return {"received": False, "error": str(e)}

# ============ ADMIN ENDPOINTS ============

async def require_admin(request: Request) -> User:
    """Helper to require admin access"""
    user = await get_current_user(request)
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    return user

@api_router.get("/admin/dashboard")
async def get_admin_dashboard(request: Request):
    """Get admin dashboard stats"""
    await require_admin(request)
    
    try:
        # Get counts
        total_users = await db.users.count_documents({})
        total_technicians = await db.users.count_documents({"role": "technician"})
        total_consumers = await db.users.count_documents({"role": "consumer"})
        total_bookings = await db.bookings.count_documents({})
        total_parts = await db.parts.count_documents({})
        
        # Get revenue stats
        transactions = await db.payment_transactions.find(
            {"payment_status": "paid"},
            {"_id": 0}
        ).to_list(10000)
        
        total_revenue = sum(t.get("amount", 0) for t in transactions)
        platform_revenue = sum(t.get("platform_commission", 0) for t in transactions)
        
        # Get recent bookings
        recent_bookings = await db.bookings.find(
            {},
            {"_id": 0}
        ).sort("created_at", -1).limit(10).to_list(10)
        
        for booking in recent_bookings:
            if isinstance(booking.get('created_at'), str):
                booking['created_at'] = datetime.fromisoformat(booking['created_at'])
        
        return {
            "stats": {
                "total_users": total_users,
                "total_technicians": total_technicians,
                "total_consumers": total_consumers,
                "total_bookings": total_bookings,
                "total_parts": total_parts,
                "total_revenue": round(total_revenue, 2),
                "platform_revenue": round(platform_revenue, 2),
                "total_transactions": len(transactions)
            },
            "recent_bookings": recent_bookings[:5]
        }
    except Exception as e:
        logging.error(f"Admin dashboard error: {e}")
        raise HTTPException(500, str(e))

@api_router.get("/admin/users", response_model=List[User])
async def get_all_users(request: Request, role: Optional[str] = None):
    """Get all users"""
    await require_admin(request)
    
    query = {}
    if role:
        query["role"] = role
    
    users = await db.users.find(query, {"_id": 0}).to_list(1000)
    
    for u in users:
        if isinstance(u.get('created_at'), str):
            u['created_at'] = datetime.fromisoformat(u['created_at'])
    
    return users

@api_router.put("/admin/users/{user_id}")
async def update_user_admin(request: Request, user_id: str, updates: Dict):
    """Admin update user"""
    await require_admin(request)
    
    await db.users.update_one(
        {"id": user_id},
        {"$set": updates}
    )
    
    return {"message": "User updated"}

@api_router.delete("/admin/users/{user_id}")
async def delete_user_admin(request: Request, user_id: str):
    """Admin delete user"""
    await require_admin(request)
    
    await db.users.delete_one({"id": user_id})
    
    return {"message": "User deleted"}

@api_router.get("/admin/bookings", response_model=List[Booking])
async def get_all_bookings(request: Request):
    """Get all bookings"""
    await require_admin(request)
    
    bookings = await db.bookings.find({}, {"_id": 0}).to_list(1000)
    
    for booking in bookings:
        if isinstance(booking.get('created_at'), str):
            booking['created_at'] = datetime.fromisoformat(booking['created_at'])
        if booking.get('completed_at') and isinstance(booking.get('completed_at'), str):
            booking['completed_at'] = datetime.fromisoformat(booking['completed_at'])
    
    return bookings

@api_router.get("/admin/transactions")
async def get_all_transactions(request: Request):
    """Get all transactions"""
    await require_admin(request)
    
    transactions = await db.payment_transactions.find({}, {"_id": 0}).to_list(1000)
    
    for t in transactions:
        if isinstance(t.get('created_at'), str):
            t['created_at'] = datetime.fromisoformat(t['created_at'])
        if t.get('updated_at') and isinstance(t.get('updated_at'), str):
            t['updated_at'] = datetime.fromisoformat(t['updated_at'])
    
    return transactions

@api_router.get("/admin/revenue-stats")
async def get_revenue_stats(request: Request):
    """Get detailed revenue statistics"""
    await require_admin(request)
    
    try:
        transactions = await db.payment_transactions.find(
            {"payment_status": "paid"},
            {"_id": 0}
        ).to_list(10000)
        
        service_transactions = [t for t in transactions if t.get("transaction_type") == "service"]
        parts_transactions = [t for t in transactions if t.get("transaction_type") == "parts"]
        
        total_revenue = sum(t.get("amount", 0) for t in transactions)
        platform_commission = sum(t.get("platform_commission", 0) for t in transactions)
        service_revenue = sum(t.get("amount", 0) for t in service_transactions)
        parts_revenue = sum(t.get("amount", 0) for t in parts_transactions)
        service_commission = sum(t.get("platform_commission", 0) for t in service_transactions)
        parts_commission = sum(t.get("platform_commission", 0) for t in parts_transactions)
        
        return {
            "total_revenue": round(total_revenue, 2),
            "platform_commission": round(platform_commission, 2),
            "service_revenue": round(service_revenue, 2),
            "parts_revenue": round(parts_revenue, 2),
            "service_commission": round(service_commission, 2),
            "parts_commission": round(parts_commission, 2),
            "service_commission_rate": f"{PLATFORM_BOOKING_COMMISSION * 100}%",
            "parts_commission_rate": f"{PLATFORM_PARTS_COMMISSION * 100}%",
            "total_transactions": len(transactions),
            "service_transactions": len(service_transactions),
            "parts_transactions": len(parts_transactions)
        }
    except Exception as e:
        logging.error(f"Revenue stats error: {e}")
        raise HTTPException(500, str(e))

@api_router.get("/admin/reviews", response_model=List[Review])
async def get_all_reviews_admin(request: Request):
    """Get all reviews"""
    await require_admin(request)
    
    reviews = await db.reviews.find({}, {"_id": 0}).to_list(1000)
    
    for review in reviews:
        if isinstance(review.get('created_at'), str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    
    return reviews

@api_router.delete("/admin/reviews/{review_id}")
async def delete_review_admin(request: Request, review_id: str):
    """Delete review"""
    await require_admin(request)
    
    await db.reviews.delete_one({"id": review_id})
    
    return {"message": "Review deleted"}

# ============ BASIC ROUTES ============

@api_router.get("/")
async def root():
    return {"message": "GearUp Repairs API"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()