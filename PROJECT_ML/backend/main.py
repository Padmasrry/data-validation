from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import motor.motor_asyncio
from bson import ObjectId
import datetime
import csv
import os
import re
import json
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.optim as optim
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import xgboost
from fuzzywuzzy import fuzz  # For fuzzy matching
import textblob  # For spell correction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock user database
mock_users = {
    "user1": "1990-01-01",
    "admin": "1985-05-15",
    "padmasrry": "2002-12-21"
}

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/csv_validation_db"
DATABASE_NAME = "csv_validation_db"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db["users"]
validation_logs_collection = db["validation_logs"]
user_feedback_collection = db["user_feedback"]

ERROR_LOG_DIR = "logs"
REPORT_DIR = "reports"

# Global variables
BERT_MODEL_PATH = "bert_intent_model.pt"
XGBOOST_MODEL_PATH = "xgboost_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
ISOLATION_MODEL_PATH = "isolation_forest_model.joblib"
vectorizer = TfidfVectorizer(max_features=500)
model = None
isolation_forest_model = None
tokenizer = None
bert_model = None

# Synonym mapping for common variations
SYNONYM_MAP = {
    "file": ["files", "document", "doc"],
    "upload": ["uploaded", "uploading", "submit"],
    "today": ["this day", "current day"],
    "what": ["whats", "wht"],
    "how": ["hw", "hows"],
    "many": ["much", "number of", "count"],
    "do": ["does", "did"],
    "application": ["app", "tool", "software"],
    "validate": ["check", "verify", "validate"],
    "status": ["state", "progress"],
    "error": ["errors", "mistake", "issue"],
    "generate": ["create", "make"],
    "report": ["pdf", "document"],
    "history": ["log", "record"],
}

def preprocess_text(text):
    """Enhanced text preprocessing without spaCy"""
    text = text.lower().strip()
    
    # Correct common misspellings
    corrections = {
        "whn": "when", "wat": "what", "wen": "when", "hw": "how",
        "pls": "please", "plz": "please", "thx": "thanks",
        "u": "you", "r": "are", "yr": "your", "wht": "what",
        "cn": "can", "whr": "where", "hws": "how's"
    }
    for wrong, right in corrections.items():
        text = re.sub(r'\b' + wrong + r'\b', right, text)
    
    # Use TextBlob for spell correction
    blob = textblob.TextBlob(text)
    corrected_text = str(blob.correct())
    
    # Replace synonyms with canonical form
    words = corrected_text.split()
    processed_words = []
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
        if word:
            replaced = False
            for canonical, synonyms in SYNONYM_MAP.items():
                if word in synonyms or word == canonical:
                    processed_words.append(canonical)
                    replaced = True
                    break
            if not replaced:
                processed_words.append(word)
    
    return " ".join(processed_words)

async def train_bert_model(questions=None, labels=None):
    global bert_model, tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if bert_model is None:
        bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 labels: general (0), greeting (1), file-related (2)
    
    # Expanded training dataset with more diverse examples
    if questions is None or labels is None:
        questions = [
            # Greetings (label 1)
            "hi", "hii", "hello", "helo", "hey", "hay", "good morning", "gd mrng", 
            "what can i help you with", "how can i hlp u", "how are you", "how r u", 
            "greetings", "greetingz", "hi there", "hii thr", "hello friend", "helo frnd",
            "good evening", "gd evng", "how can i assist you today", "how i assist u tdy", 
            "hey buddy", "hi mate", "hello there", "good afternoon", "gd aftrnoon", 
            "what’s up", "whats up", "howdy", "gday", "yo", "hola", "salut",
            # File-related (label 2)
            "how many files uploaded today", "how much file upload tdy", 
            "how many files uploaded 19-04-2025", "file upload 19-04-2025",
            "file count for march 2025", "files cnt march 2025", "show errors for id 123", 
            "show err id 123", "display errors for file 456", "display err file 456", 
            "get errors id 789", "get err id 789", "validate my data", "check my data plz", 
            "run validation now", "validate now", "data validation please", 
            "predict issues for id 456", "predict issue id 456", "forecast problems id 123", 
            "forecast prob id 123", "issue prediction for 789", "predict issue 789", 
            "generate report for id 789", "create report id 789", "export report for 456", 
            "export rprt 456", "can i upload a csv file", "can i upload csv", 
            "is csv upload supported", "csv upload support", "upload csv allowed", 
            "can upload csv?", "where can i see validation history", "where validation log", 
            "validation history location", "how do i correct errors found", "how fix error", 
            "error correction steps", "fix error step", "how do i start a new validation", 
            "start new validate", "begin new validation", "start validation now", 
            "show validation status for id 101", "status id 101", "status for file 2", 
            "check status id 303", "validation status with id 404", "status id 404", 
            "file status 505", "status of id 606", "analyze data for id 707", 
            "data analysis id 808", "review data 909", "suggest fixes for id 1010", 
            "fix suggestions 1111", "correct issues 1212", "download report id 1313", 
            "export pdf 1414", "get report 1515", "feedback yes for id 1616", 
            "feedback no id 1717", "rate this 1819", "list uploaded files", 
            "list all file", "all files today", "recent uploads", "was last year busier",
            "growth this year", "most active month", "most activ mnth",
            "files in january 2025", "how many files last month", "files last month",
            # General (label 0)
            "what does this application do", "what this app do", "what is this app for", 
            "what this app 4", "tell me about this app", "tell about app", 
            "is my data secure", "data secure?", "data security info", 
            "is data safe here", "data safe?", "what file formats are supported", 
            "which file format support", "supported file types", "list file formats", 
            "list file type", "how does this work", "explain the tool", 
            "what is data validation", "tell me about validation"
        ]
        labels = [1] * 35 + [2] * 93 + [0] * 20  # 35 greetings, 93 file-related, 20 general
    
    # Ensure questions and labels have the same length
    min_length = min(len(questions), len(labels))
    questions = questions[:min_length]
    labels = labels[:min_length]
    
    # Data augmentation: Add slight variations
    augmented_questions = questions.copy()
    augmented_labels = labels.copy()
    for q, l in zip(questions, labels):
        if "file" in q:
            augmented_questions.append(q.replace("file", "document"))
            augmented_labels.append(l)
        if "upload" in q:
            augmented_questions.append(q.replace("upload", "submit"))
            augmented_labels.append(l)
    
    # Prepare inputs
    inputs = tokenizer(augmented_questions, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels_tensor = torch.tensor(augmented_labels)
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    indices = list(range(len(augmented_questions)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=augmented_labels)
    
    train_inputs = {key: inputs[key][train_idx] for key in inputs}
    test_inputs = {key: inputs[key][test_idx] for key in inputs}
    train_labels = labels_tensor[train_idx]
    test_labels = labels_tensor[test_idx]
    
    # Optimizer and training
    optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
    bert_model.train()
    for epoch in range(10):  # Increased epochs for better learning
        outputs = bert_model(**train_inputs, labels=train_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}/10, Train Loss: {loss.item()}")
    
    # Evaluate
    bert_model.eval()
    with torch.no_grad():
        test_outputs = bert_model(**test_inputs, labels=test_labels)
        predictions = torch.argmax(test_outputs.logits, dim=1)
        accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
        print(f"Test Accuracy: {accuracy:.4f}")
    
    torch.save(bert_model.state_dict(), BERT_MODEL_PATH)
    print(f"Trained and saved BERT model to {BERT_MODEL_PATH} (size: {os.path.getsize(BERT_MODEL_PATH)} bytes)")

async def train_xgboost_model(questions=None, labels=None):
    # Placeholder to maintain compatibility
    print("XGBoost training skipped; using BERT model only.")
    global model, vectorizer
    model = None
    vectorizer = None
    joblib.dump(None, XGBOOST_MODEL_PATH)
    joblib.dump(None, VECTORIZER_PATH)

async def train_isolation_forest():
    global isolation_forest_model
    try:
        logs = await validation_logs_collection.find().limit(10).to_list(None)
        all_data = []
        for log in logs:
            upload_path = os.path.join("uploads", log["file_name"])
            if os.path.exists(upload_path):
                df = pd.read_csv(upload_path, dtype=str)
                numeric_cols = ["group", "ean", "weight", "price"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        all_data.extend(df[col].dropna().values)
        if not all_data:
            np.random.seed(42)
            normal_data = np.random.normal(loc=50, scale=5, size=(800, 1))
            anomalous_data = np.random.normal(loc=100, scale=10, size=(200, 1))
            all_data = np.vstack([normal_data, anomalous_data])
            labels = np.array([1] * 800 + [-1] * 200)
        else:
            all_data = np.array(all_data).reshape(-1, 1)
            labels = np.ones(len(all_data))
            anomaly_threshold = int(len(all_data) * 0.9)
            labels[anomaly_threshold:] = -1

        isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest_model.fit(all_data)
        joblib.dump(isolation_forest_model, ISOLATION_MODEL_PATH)
        return isolation_forest_model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global model, vectorizer, isolation_forest_model, bert_model, tokenizer
    if not os.path.exists(XGBOOST_MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        await train_xgboost_model()
    else:
        model = joblib.load(XGBOOST_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Loaded pre-trained XGBoost model")
    if not os.path.exists(ISOLATION_MODEL_PATH):
        isolation_forest_model = await train_isolation_forest()
    else:
        isolation_forest_model = joblib.load(ISOLATION_MODEL_PATH)
        print("Loaded pre-trained IsolationForest model")
    if not os.path.exists(BERT_MODEL_PATH):
        print("BERT model not found. Training new model...")
        await train_bert_model()
    else:
        print("Loading pre-trained BERT model...")
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
            state_dict = torch.load(BERT_MODEL_PATH, weights_only=False)
            bert_model.load_state_dict(state_dict)
            bert_model.eval()
            print("Loaded pre-trained BERT model")
        except Exception as e:
            print(f"Error loading BERT model: {e}. Retraining model...")
            await train_bert_model()

async def retrain_models():
    global bert_model, tokenizer
    feedback = await user_feedback_collection.find().to_list(None)
    new_questions = []
    new_labels = []
    new_responses = []

    for f in feedback:
        if "response" in f and f.get("intent") is not None and f["response"].strip():
            question = f["question"].lower().strip()
            if "I didn’t get that" in f["response"]:
                continue
            new_questions.append(question)
            new_labels.append(f["intent"])
            new_responses.append(f["response"])

    if not new_questions:
        print("No new feedback data to retrain with. Using initial data.")
        await train_bert_model()
        return

    # Load initial dataset
    initial_questions = [
        "hi", "hii", "hello", "hey", "good morning", "what does this application do?", 
        "what this app do", "how many files today?", "how much file tdy", 
        "list all uploaded files", "show errors for id 123", "validation status for id 101", 
        "status id 101"
    ]
    initial_labels = [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2]
    all_questions = initial_questions + new_questions
    all_labels = initial_labels + new_labels

    # Balance the dataset
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in all_labels:
        label_counts[label] += 1
    max_count = max(label_counts.values())
    balanced_questions = []
    balanced_labels = []
    for label in range(3):
        label_indices = [i for i, l in enumerate(all_labels) if l == label]
        label_questions = [all_questions[i] for i in label_indices]
        label_labels = [all_labels[i] for i in label_indices]
        repeat_factor = max_count // len(label_questions) if len(label_questions) > 0 else 1
        balanced_questions.extend(label_questions * repeat_factor)
        balanced_labels.extend(label_labels * repeat_factor)

    # Data augmentation
    augmented_questions = balanced_questions.copy()
    augmented_labels = balanced_labels.copy()
    for q, l in zip(balanced_questions, balanced_labels):
        if "file" in q:
            augmented_questions.append(q.replace("file", "document"))
            augmented_labels.append(l)
        if "upload" in q:
            augmented_questions.append(q.replace("upload", "submit"))
            augmented_labels.append(l)

    # Prepare BERT inputs
    inputs = tokenizer(augmented_questions, padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels_tensor = torch.tensor(augmented_labels)
    
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(np.arange(len(augmented_labels)), test_size=0.2, random_state=42, stratify=augmented_labels)
    train_inputs = {key: inputs[key][train_indices] for key in inputs}
    test_inputs = {key: inputs[key][test_indices] for key in inputs}
    train_labels = labels_tensor[train_indices]
    test_labels = labels_tensor[test_indices]

    # Retrain BERT
    optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
    bert_model.train()
    for epoch in range(10):
        outputs = bert_model(**train_inputs, labels=train_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Retraining BERT - Epoch {epoch + 1}/10, Train Loss: {loss.item()}")
    bert_model.eval()
    with torch.no_grad():
        test_outputs = bert_model(**test_inputs, labels=test_labels)
        predictions = torch.argmax(test_outputs.logits, dim=1)
        accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
        print(f"Retraining BERT - Test Accuracy: {accuracy:.4f}")
    torch.save(bert_model.state_dict(), BERT_MODEL_PATH)

    # Update response_map
    global response_map
    for q, r in zip(new_questions, new_responses):
        if q not in response_map and r and "I didn’t get that" not in r:
            response_map[q] = r

    print("BERT model retrained and response_map updated successfully.")

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str
    @validator('password')
    def validate_dob(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError('Password must be in YYYY-MM-DD format')
        return v

class RegisterRequest(BaseModel):
    empId: str
    name: str
    dob: str
    companyBranch: str
    designation: str
    email: str
    phone: str
    @validator('dob')
    def validate_dob(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError('DOB must be in YYYY-MM-DD format')
        return v
    @validator('email')
    def validate_email(cls, v):
        if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError('Invalid email format')
        return v
    @validator('phone')
    def validate_phone(cls, v):
        if not re.match(r"^\d{10}$", v):
            raise ValueError('Phone number must be 10 digits')
        return v

class ChatRequest(BaseModel):
    question: str
    file_id: str = None
    empId: str = None

def get_empId(empId: str = Depends(lambda: None)):
    return empId or "padmasrry"

# Validation Rules
VALIDATION_RULES = {
    "product": {"type": str, "max_length": 25, "mandatory": True, "unique": True},
    "name": {"type": str, "max_length": 250, "mandatory": True},
    "group": {"type": int, "max_length": 10, "mandatory": True},
    "ean": {"type": int, "max_length": 12},
    "weight": {"type": float},
    "inventory_unit": {"type": str, "allowed_values": ["EA", "KG"]},
    "price": {"type": float},
    "category": {"type": str},
}

def validate_row(row, column, rules):
    value = row.get(column)
    if value is None:
        return not rules.get("mandatory", False)
    if rules.get("type") == int:
        try: int(value)
        except ValueError: return False
    elif rules.get("type") == float:
        try: float(value)
        except ValueError: return False
    if "max_length" in rules and isinstance(value, str) and len(value) > rules["max_length"]:
        return False
    if "allowed_values" in rules and value not in rules["allowed_values"]:
        return False
    if "regex" in rules and isinstance(value, str) and not re.match(rules["regex"], value):
        return False
    return True

async def get_next_unique_id():
    last_entry = await validation_logs_collection.find_one(sort=[("unique_id", -1)])
    return (last_entry["unique_id"] + 1) if last_entry else 1

@app.post("/retrain/")
async def trigger_retrain():
    await retrain_models()
    return {"message": "Models retrained successfully"}

@app.post("/register/")
async def register(request: RegisterRequest):
    existing_user = await users_collection.find_one({"empId": request.empId})
    if existing_user:
        return {"success": False, "message": "Employee ID already exists"}
    user_data = request.dict()
    user_data["createdAt"] = datetime.datetime.now().isoformat()
    await users_collection.insert_one(user_data)
    mock_users[request.empId] = request.dob
    return {"success": True, "message": "Registration successful"}

@app.post("/login/")
async def login(request: LoginRequest):
    username_lower = request.username.lower()
    user = await users_collection.find_one({
        "$or": [
            {"empId": request.username},
            {"name": username_lower}
        ]
    })
    if user and user["dob"] == request.password:
        return {"success": True, "message": "Login successful", "empId": user["empId"]}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...), empId: str = Depends(get_empId)):
    start_time = datetime.datetime.now()
    unique_id = await get_next_unique_id()
    
    try:
        df = pd.read_csv(file.file, dtype=str)
        print(f"Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"CSV read error: {e}")
        return {"error": f"Failed to read CSV: {str(e)}"}

    errors = []
    unique_values = {}
    anomaly_results = []

    for index, row in df.iterrows():
        for column, rules in VALIDATION_RULES.items():
            value = row[column] if column in row else None
            if rules.get("type") == int:
                try: value = int(value) if value else None
                except ValueError: errors.append([index, column, "Expected integer (Critical)"])
            elif rules.get("type") == float:
                try: value = float(value) if value else None
                except ValueError: errors.append([index, column, "Expected float (Warning)"])
            if rules.get("mandatory") and (value is None or str(value).strip() == ""):
                errors.append([index, column, "Missing mandatory value (Critical)"])
            if "max_length" in rules and isinstance(value, str) and len(value) > rules["max_length"]:
                errors.append([index, column, f"Exceeds max length of {rules['max_length']} (Warning)"])
            if "allowed_values" in rules and value not in rules["allowed_values"]:
                errors.append([index, column, f"Invalid value. Allowed: {rules['allowed_values']} (Critical)"])
            if "regex" in rules and isinstance(value, str) and not re.match(rules["regex"], value):
                errors.append([index, column, "Invalid format (Warning)"])
            if rules.get("unique"):
                if column not in unique_values:
                    unique_values[column] = set()
                if value in unique_values[column]:
                    errors.append([index, column, "Duplicate value (Critical)"])
                unique_values[column].add(value)

    numeric_columns = ["group", "ean", "weight", "price"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            data = df[column].dropna().values.reshape(-1, 1)
            if len(data) > 0 and isolation_forest_model:
                predictions = isolation_forest_model.predict(data)
                anomalies = data[predictions == -1].flatten().tolist()
                data_indices = df[column].dropna().index
                anomaly_indices = data_indices[predictions == -1]
                for idx, anomaly_value in zip(anomaly_indices, anomalies):
                    anomaly_results.append([int(idx), column, f"Anomaly detected: {anomaly_value}"])

    combined_errors = errors + anomaly_results

    error_dir = "logs"
    os.makedirs(error_dir, exist_ok=True)
    error_filename = f"error_log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    error_filepath = os.path.join(error_dir, error_filename)

    with open(error_filepath, mode="w", newline="") as error_file:
        writer = csv.writer(error_file)
        writer.writerow(["Row", "Column", "Error Message"])
        writer.writerows(combined_errors)

    end_time = datetime.datetime.now()
    time_taken = (end_time - start_time).total_seconds()

    validation_log = {
        "unique_id": unique_id,
        "file_name": file.filename,
        "total_errors": len(errors),
        "total_anomalies": len(anomaly_results),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "time_taken": time_taken,
        "error_log_file": error_filename,
        "uploaded_by": empId
    }
    await validation_logs_collection.insert_one(validation_log)

    return {
        "message": "File processed successfully",
        "unique_id": unique_id,
        "errors": len(errors),
        "anomalies": len(anomaly_results),
        "error_log": error_filename
    }

@app.get("/get_validation_logs/")
async def get_validation_logs(empId: str = Depends(get_empId)):
    logs = await validation_logs_collection.find({"uploaded_by": empId}).sort("start_time", -1).limit(10).to_list(None)
    response = [
        {
            "unique_id": log["unique_id"],
            "file_name": log["file_name"],
            "total_errors": log["total_errors"],
            "total_anomalies": log.get("total_anomalies", 0),
            "time_taken": log["time_taken"],
            "start_time": log["start_time"],
            "end_time": log["end_time"],
            "error_log_file": log["error_log_file"],
        }
        for log in logs
    ]
    return response

@app.get("/get_dashboard_metrics/")
async def get_dashboard_metrics(empId: str = Depends(get_empId), period: str = "30"):
    print("Fetching dashboard metrics at:", datetime.datetime.now())
    total_files = await validation_logs_collection.count_documents({"uploaded_by": empId})
    now = datetime.datetime.now()
    period_days = int(period)
    period_ago = now - datetime.timedelta(days=period_days)
    monthly_logs = await validation_logs_collection.find({"uploaded_by": empId, "start_time": {"$gte": period_ago.isoformat()}}).to_list(None)
    print("Monthly logs fetched:", len(monthly_logs), "records")
    total_monthly_runs = len(monthly_logs)
    average_monthly_runs = total_monthly_runs / period_days if total_monthly_runs > 0 else 0.0

    monthly_data = {}
    for log in monthly_logs:
        start_time = datetime.datetime.fromisoformat(log["start_time"])
        month_year = start_time.strftime("%Y-%m")
        if month_year not in monthly_data:
            monthly_data[month_year] = {"file_count": 0, "total_runs": 0, "total_anomalies": 0}
        monthly_data[month_year]["file_count"] += 1
        monthly_data[month_year]["total_runs"] += 1
        monthly_data[month_year]["total_anomalies"] += log.get("total_anomalies", 0)

    monthly_breakdown = [
        {
            "month": month,
            "file_count": data["file_count"],
            "average_runs": data["total_runs"] / (period_days / 30) if period_days >= 30 else data["total_runs"],
            "total_anomalies": data["total_anomalies"]
        }
        for month, data in monthly_data.items()
    ]

    recent_logs = await validation_logs_collection.find({"uploaded_by": empId}).sort("start_time", -1).limit(5).to_list(None)
    print("Recent logs fetched:", len(recent_logs), "records")
    recent_activity = [
        f"Uploaded {log['file_name']} on {datetime.datetime.fromisoformat(log['start_time']).strftime('%m/%d/%Y, %I:%M:%S %p')} with {log['total_errors']} errors and {log.get('total_anomalies', 0)} anomalies"
        for log in recent_logs
    ]
    print("Dashboard metrics processed at:", datetime.datetime.now())
    
    return {
        "total_files": total_files,
        "average_monthly_runs": average_monthly_runs,
        "recent_activity": recent_activity,
        "monthly_breakdown": monthly_breakdown,
    }

@app.get("/download_csv/{file_name}")
async def download_csv(file_name: str, empId: str = Depends(get_empId)):
    file_path = os.path.join(ERROR_LOG_DIR, file_name)
    if os.path.exists(file_path):
        try:
            return FileResponse(file_path, media_type="text/csv", filename=file_name)
        except Exception as e:
            return {"error": f"An error occurred while serving the file: {str(e)}"}
    else:
        return {"error": "File not found"}

# Initialize response_map
response_map = {
    "how many files did i upload this month?": "Let me check... I'll count the files you uploaded this month.",
    "what are the files uploaded by user padmasrry?": "Let me fetch the files uploaded by Padmasrry for you.",
    "what are the files uploaded by padmasrry?": "Let me fetch the files uploaded by Padmasrry for you.",
    "what does this application do?": "This application helps you validate CSV files by checking for errors, duplicates, and anomalies.",
    "how can i validate my data here?": "Upload a CSV file via the /upload_csv/ endpoint or use the interface to start validation.",
    "is this tool free to use?": "Yes, this tool is free with limited usage.",
    "what kind of data can i validate?": "You can validate CSV files with columns like product, name, group, ean, weight, price, etc.",
    "is my uploaded data secure?": "Your data is handled securely, but consult the privacy policy for details.",
    "how do i upload a file?": "Use the /upload_csv/ endpoint with a POST request and a file attachment.",
    "what file formats are supported?": "Currently, only CSV files are supported.",
    "can i upload an excel file?": "No, only CSV files are supported currently.",
    "is there a file size limit?": "There’s no strict limit, but large files may take longer to process.",
    "what should the column headers be?": "Headers should match validation rules like product, name, group, etc.",
    "can i check for duplicate records?": "Yes, the tool checks for duplicates in unique columns.",
    "how do i start a new validation?": "Upload a new CSV file via /upload_csv/.",
    "can i save my settings for future use?": "No, settings are not saved yet.",
    "where can i see the error summary?": "Check the error log file after validation.",
    "how do i download the cleaned data?": "Cleaning is not automated; download the original with fixes applied manually.",
    "can i generate a pdf report?": "Yes, generate a PDF report via /generate report/.",
    "is my data stored on your server?": "Yes, temporarily for processing.",
    "how long is my data retained?": "Data is retained for 30 days unless deleted.",
    "can you help me validate this file?": "Yes, upload the file via /upload_csv/ and I’ll assist.",
    "where can i find the validation history?": "Use /get_validation_logs/ to see your history."
}

@app.post("/chat/")
async def chat(request: ChatRequest, empId: str = Depends(get_empId)):
    global tokenizer, bert_model, response_map
    logs = await validation_logs_collection.find({"uploaded_by": empId}).to_list(None)
    recent_log = await validation_logs_collection.find_one({"unique_id": int(request.file_id)}) if request.file_id and request.file_id.isdigit() else None
    
    if bert_model is None or tokenizer is None:
        print("BERT model not initialized. Training model...")
        await train_bert_model()

    # Preprocess the user's question
    original_question = request.question
    processed_question = preprocess_text(original_question)
    print(f"Original question: {original_question}")
    print(f"Processed question: {processed_question}")

    # Check response_map with fuzzy matching
    best_match = None
    best_score = 0
    for q in response_map:
        score = fuzz.ratio(processed_question.lower(), q.lower())
        if score > 80 and score > best_score:
            best_match = q
            best_score = score

    if best_match:
        if best_match.lower() == "how many files did i upload this month?":
            today = datetime.datetime.now()
            start_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_files = [log for log in logs if start_month <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) <= today]
            print(f"Fuzzy response_map match - Files this month for empId {empId}: {len(month_files)}")
            response = f"You uploaded {len(month_files)} files this month."
        elif best_match.lower() in ["what are the files uploaded by user padmasrry?", "what are the files uploaded by padmasrry?"]:
            target_user = "padmasrry"
            user_logs = await validation_logs_collection.find({"uploaded_by": target_user}).to_list(None)
            file_list = [f"ID: {log['unique_id']} - {log['file_name']} (Uploaded: {log['start_time']})" for log in user_logs]
            response = f"Files uploaded by {target_user}:\n" + "\n".join(file_list) if file_list else f"No files uploaded by {target_user}."
        else:
            response = response_map[best_match]
        await user_feedback_collection.insert_one({
            "empId": empId,
            "question": original_question.lower(),
            "processed_question": processed_question,
            "response": response,
            "intent": None,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {"response": response}

    # Classify intent: 0 (general), 1 (greeting), 2 (file-related)
    inputs = tokenizer(processed_question, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=1)
        predicted_intent = torch.argmax(prediction, dim=1).item()
        print(f"Predicted intent for '{processed_question}': {predicted_intent} (Probabilities: {prediction.tolist()})")

    # Log the interaction
    await user_feedback_collection.insert_one({
        "empId": empId,
        "question": original_question.lower(),
        "processed_question": processed_question,
        "response": "",
        "intent": int(predicted_intent),
        "timestamp": datetime.datetime.now().isoformat()
    })

    # Handle empty input
    if not original_question.strip():
        response = "Hey, I am the chatbot separately created for this data validation. You can ask me anything about the data here!"
        await user_feedback_collection.update_one(
            {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
            {"$set": {"response": response}}
        )
        return {"response": response}

    # Handle greetings (intent 1)
    if predicted_intent == 1:
        response = "Hi! What can I help you with?"
        await user_feedback_collection.update_one(
            {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
            {"$set": {"response": response}}
        )
        return {"response": response}

    # Handle file-related (intent 2)
    today = datetime.datetime.now()
    print(f"Number of logs for empId {empId}: {len(logs)}")
    match = re.search(r'(today|yesterday|this\s+week|(\d{1,2})\s+(?:of\s+)?(\w+)(?:\s+(\d{4}))?|last\s+year|past\s+3\s+months|(\d{4}))', processed_question, re.IGNORECASE)
    target_date = None
    year = None
    month = None
    if match:
        if match.group(1) == "today":
            target_date = today
        elif match.group(1) == "yesterday":
            target_date = today - datetime.timedelta(days=1)
        elif match.group(1) == "this week":
            target_date = today - datetime.timedelta(days=today.weekday())
        elif match.group(2) and match.group(3):
            month_name = match.group(3).lower()
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            month = month_map.get(month_name)
            day = int(match.group(2)) if match.group(2) else 1
            year = int(match.group(4)) if match.group(4) else today.year
            if not year:
                response = f"Please specify a year with {day} {month_name} (e.g., 2024)."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            target_date = datetime.datetime(year, month, day)
        elif match.group(1) == "last year":
            target_date = today.replace(year=today.year - 1)
        elif match.group(1) == "past 3 months":
            target_date = today - datetime.timedelta(days=90)
        elif match.group(5):
            year = int(match.group(5))
            target_date = datetime.datetime(year, 1, 1)

    if predicted_intent == 2:
        if "how many files" in processed_question or "number of files" in processed_question:
            if target_date:
                start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + datetime.timedelta(days=1) if target_date.date() == today.date() else (
                    start_date.replace(month=start_date.month + 1) if start_date.month < 12 else 
                    start_date.replace(year=start_date.year + 1, month=1)
                )
                monthly_files = [
                    log for log in logs if start_date <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < end_date
                ]
                actual_count = len(monthly_files)
                if year and month and day:
                    response = f"{actual_count} files are uploaded on {target_date.strftime('%B %d, %Y')}."
                elif year and month:
                    response = f"{actual_count} files are uploaded in {target_date.strftime('%B %Y')}."
                elif year:
                    response = f"{actual_count} files are uploaded in {year}."
                elif "this week" in processed_question:
                    response = f"{actual_count} files are uploaded this week."
                elif "past 3 months" in processed_question:
                    response = f"{actual_count} files are uploaded in the past 3 months."
                else:
                    response = f"{actual_count} files are uploaded today."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            response = "Please specify a date, month, year, or period (e.g., 'today', 'March 2024', 'past 3 months')."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "was" in processed_question and ("higher" in processed_question or "busier" in processed_question):
            if "last year" in processed_question and "this year" in processed_question:
                last_year_start = today.replace(year=today.year - 1, month=1, day=1)
                this_year_start = today.replace(month=1, day=1)
                last_year_files = len([log for log in logs if last_year_start <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < last_year_start.replace(year=last_year_start.year + 1)])
                this_year_files = len([log for log in logs if this_year_start <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < today])
                response = "Yes" if last_year_files > this_year_files else "No"
                response = f"Was last year's usage higher than this year? {response}."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            elif match.group(2) and match.group(3) and "than" in processed_question:
                month1 = month_map.get(match.group(3).lower())
                year1 = int(match.group(4)) if match.group(4) else today.year
                start1 = datetime.datetime(year1, month1, 1)
                end1 = start1.replace(month=month1 + 1) if month1 < 12 else start1.replace(year=year1 + 1, month=1)
                files1 = len([log for log in logs if start1 <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < end1])
                next_match = re.search(r'than\s+(\w+)', processed_question)
                if next_match:
                    month2_name = next_match.group(1)
                    month2 = month_map.get(month2_name)
                    start2 = datetime.datetime(year1, month2, 1)
                    end2 = start2.replace(month=month2 + 1) if month2 < 12 else start2.replace(year=year1 + 1, month=1)
                    files2 = len([log for log in logs if start2 <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < end2])
                    response = "Yes" if files1 > files2 else "No"
                    response = f"Was {match.group(3).capitalize()} busier than {month2_name.capitalize()}? {response}."
                    await user_feedback_collection.update_one(
                        {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                        {"$set": {"response": response}}
                    )
                    return {"response": response}
            response = "Please specify years or months to compare (e.g., 'Was last year higher than this year?')."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "most active month" in processed_question or "growth in validations" in processed_question:
            if "last year" in processed_question:
                last_year_start = today.replace(year=today.year - 1, month=1, day=1)
                monthly_counts = {}
                for log in logs:
                    log_date = datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00'))
                    if last_year_start <= log_date < last_year_start.replace(year=last_year_start.year + 1):
                        month_key = log_date.strftime('%B %Y')
                        monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                most_active = max(monthly_counts.items(), key=lambda x: x[1]) if monthly_counts else ("None", 0)
                response = f"The most active month last year was {most_active[0]} with {most_active[1]} files."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            elif "from last year to this year" in processed_question:
                last_year_start = today.replace(year=today.year - 1, month=1, day=1)
                this_year_start = today.replace(month=1, day=1)
                last_year_files = len([log for log in logs if last_year_start <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < last_year_start.replace(year=last_year_start.year + 1)])
                this_year_files = len([log for log in logs if this_year_start <= datetime.datetime.fromisoformat(log['start_time'].replace('Z', '+00:00')) < today])
                growth = ((this_year_files - last_year_files) / last_year_files * 100) if last_year_files > 0 else 100 if this_year_files > 0 else 0
                response = f"Growth in validations from last year to this year: {growth:.1f}%."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            response = "Please specify a year or period (e.g., 'most active month last year')."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "what are the files uploaded by" in processed_question:
            user_match = re.search(r'what are the files uploaded by (?:user\s+)?(\w+)', processed_question)
            if user_match:
                target_user = user_match.group(1)
                user_logs = await validation_logs_collection.find({"uploaded_by": target_user}).to_list(None)
                file_list = [f"ID: {log['unique_id']} - {log['file_name']} (Uploaded: {log['start_time']})" for log in user_logs]
                response = f"Files uploaded by {target_user}:\n" + "\n".join(file_list) if file_list else f"No files uploaded by {target_user}."
            else:
                response = "Please specify a username (e.g., 'What are the files uploaded by user Padmasrry?')."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "validation status" in processed_question and "id" in processed_question:
            file_id = re.search(r'\d+', processed_question)
            if file_id and (recent_log := await validation_logs_collection.find_one({"unique_id": int(file_id.group())})):
                status = "Successful" if recent_log["total_errors"] == 0 else f"Failed with {recent_log['total_errors']} errors and {recent_log.get('total_anomalies', 0)} anomalies"
                response = f"Validation Status for file ID {file_id.group()}: {status}, processed in {recent_log['time_taken']:.2f} seconds"
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            response = "Couldn’t find that file ID. Please provide a valid ID."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "list uploaded files" in processed_question or "all files" in processed_question:
            file_list = [f"ID: {log['unique_id']} - {log['file_name']} (Uploaded: {log['start_time']})" for log in logs]
            response = "Your uploaded files:\n" + "\n".join(file_list) if file_list else "No files uploaded yet."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "show errors" in processed_question and "id" in processed_question:
            file_id = re.search(r'\d+', processed_question)
            if file_id and (recent_log := await validation_logs_collection.find_one({"unique_id": int(file_id.group())})):
                total_errors = recent_log["total_errors"]
                response = f"Total errors for file ID {file_id.group()}: {total_errors}"
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response}
            response = "No errors found or invalid ID provided."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "suggest fixes" in processed_question and "id" in processed_question:
            file_id = re.search(r'\d+', processed_question)
            if file_id and (recent_log := await validation_logs_collection.find_one({"unique_id": int(file_id.group())})) and (recent_log["total_errors"] > 0 or recent_log.get("total_anomalies", 0) > 0):
                fixes = []
                error_file = os.path.join(ERROR_LOG_DIR, recent_log["error_log_file"])
                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        next(f)
                        for row in csv.reader(f):
                            if "Missing mandatory value" in row[2]:
                                fixes.append(f"Row {row[0]} - Set '{row[1]}' to 'Unknown'")
                            elif "Invalid value" in row[2]:
                                fixes.append(f"Row {row[0]} - Correct '{row[1]}' to a valid value")
                response = "Suggested Fixes:\n" + "\n".join(fixes) if fixes else "No fixes needed."
                await user_feedback_collection.update_one(
                    {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                    {"$set": {"response": response}}
                )
                return {"response": response, "apply": f"Apply fixes for ID {file_id.group()}? (Yes/No)"}
            response = "No errors to suggest fixes for or invalid ID."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

        elif "generate report" in processed_question and "id" in processed_question:
            file_id = re.search(r'\d+', processed_question)
            if file_id and (recent_log := await validation_logs_collection.find_one({"unique_id": int(file_id.group())})):
                try:
                    upload_path = os.path.join("uploads", recent_log["file_name"])
                    df = pd.read_csv(upload_path, dtype=str) if os.path.exists(upload_path) else None
                    if df is not None:
                        report_path = generate_report(recent_log, df)
                        response = f"Report generated for ID {file_id.group()}"
                        await user_feedback_collection.update_one(
                            {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                            {"$set": {"response": response}}
                        )
                        return {"response": response, "download": f"/download_report/{os.path.basename(report_path)}"}
                except Exception as e:
                    response = f"Error generating report: {str(e)}"
                    await user_feedback_collection.update_one(
                        {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                        {"$set": {"response": response}}
                    )
                    return {"response": response}
            response = "No report generated or invalid ID."
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

    # Handle general (intent 0) or fallback
    for q in response_map:
        score = fuzz.ratio(processed_question.lower(), q.lower())
        if score > 80:
            response = response_map[q]
            await user_feedback_collection.update_one(
                {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
                {"$set": {"response": response}}
            )
            return {"response": response}

    # Fallback for unrecognized questions
    recent_commands = [log["question"] for log in await user_feedback_collection.find({"empId": empId}).sort("timestamp", -1).limit(3).to_list(None)]
    suggested_questions = recent_commands if recent_commands else ["how many files today?", "what does this application do?"]
    response = f"I didn’t get that. Try asking something like: {', '.join(suggested_questions[:2])}. Or tell me more!"
    await user_feedback_collection.update_one(
        {"timestamp": {"$gt": (datetime.datetime.now() - datetime.timedelta(seconds=1)).isoformat()}},
        {"$set": {"response": response}}
    )
    return {"response": response}

def generate_report(log, df):
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"report_{log['unique_id']}.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    story = []

    story.append(Paragraph(f"Validation Report - File ID {log['unique_id']}"))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"File Name: {log['file_name']}"))
    story.append(Paragraph(f"Status: {'Successful' if log['total_errors'] == 0 else f'Failed with {log['total_errors']} errors and {log.get('total_anomalies', 0)} anomalies'}"))
    story.append(Paragraph(f"Processed: {log['start_time']} - {log['end_time']} ({log['time_taken']:.2f} seconds)"))

    if log['total_errors'] > 0 or log.get("total_anomalies", 0) > 0:
        error_file = os.path.join(ERROR_LOG_DIR, log["error_log_file"])
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                next(f)
                errors = list(csv.reader(f))
                story.append(Paragraph("Errors/Anomalies:"))
                story.append(Paragraph(str(errors)))

    numeric_cols = [col for col in ["weight", "price", "group"] if col in df.columns]
    if numeric_cols:
        analysis = {col: {"mean": df[col].mean(), "std": df[col].std()} for col in numeric_cols if df[col].dtype in [np.float64, np.int64]}
        story.append(Paragraph("Data Analysis:"))
        story.append(Paragraph(str(analysis)))

    doc.build(story)
    return report_path

@app.get("/download_report/{file_name}")
async def download_report(file_name: str, empId: str = Depends(get_empId)):
    file_path = os.path.join(REPORT_DIR, file_name)
    if os.path.exists(file_path):
        try:
            return FileResponse(file_path, media_type="application/pdf", filename=file_name)
        except Exception as e:
            return {"error": f"An error occurred while serving the file: {str(e)}"}
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)