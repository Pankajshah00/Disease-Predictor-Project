from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fuzzywuzzy import process
from textblob import TextBlob
import numpy as np
import random
import os
import json
import requests # Import the requests library

app = Flask(__name__)
app.config['SECRET_KEY'] = 'YOUR_SUPER_SECRET_KEY_HERE_REPLACE_ME_IN_PRODUCTION!'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    prediction_history_json = db.Column(db.Text, default='[]')

    def __init__(self, username, password):
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.prediction_history_json = '[]'

    def get_id(self):
        return str(self.id)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_prediction_history(self):
        return json.loads(self.prediction_history_json)

    def add_prediction_to_history(self, entry):
        history = self.get_prediction_history()
        history.append(entry)
        self.prediction_history_json = json.dumps(history)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

CSV_DIR = 'datasets/'

try:
    data = pd.read_csv(os.path.join(CSV_DIR, 'Training.csv'))
    description_df = pd.read_csv(os.path.join(CSV_DIR, 'description.csv'))
    precautions_df = pd.read_csv(os.path.join(CSV_DIR, 'precautions_df.csv'))
    medications_df = pd.read_csv(os.path.join(CSV_DIR, 'medications.csv'))
    diets_df = pd.read_csv(os.path.join(CSV_DIR, 'diets.csv'))
    workouts_df = pd.read_csv(os.path.join(CSV_DIR, 'workout_df.csv'))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Missing CSV file - {e.filename}. Please ensure all required CSVs are in the '{CSV_DIR}' directory relative to main.py")
    print("Application cannot start without these files. Exiting.")
    exit()

X = data.drop('prognosis', axis=1)
y = data['prognosis']

symptom_names_raw = [col.lower().replace('_', ' ') for col in X.columns]
symptom_index = {symptom.replace(' ', '_'): i for i, symptom in enumerate(symptom_names_raw)}
valid_symptom_list = list(symptom_names_raw)

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

disease_info = {}
for index, row in description_df.iterrows():
    disease_lower = row['Disease'].lower()
    disease_info[disease_lower] = {
        'description': row['Description'],
        'precautions': precautions_df[precautions_df['Disease'].str.lower() == disease_lower].drop('Disease', axis=1).values.flatten().tolist(),
        'medications': medications_df[medications_df['Disease'].str.lower() == disease_lower]['Medication'].tolist() if 'Medication' in medications_df.columns else [],
        'diets': diets_df[diets_df['Disease'].str.lower() == disease_lower]['Diet'].tolist() if 'Diet' in diets_df.columns else [],
        'workouts': workouts_df[workouts_df['Disease'].str.lower() == disease_lower]['Workout'].tolist() if 'Workout' in workouts_df.columns else [],
        'emergency': False
    }

emergency_diseases = [
    "aids", "stroke", "heart attack", "dengue", "malaria", "typhoid",
    "pneumonia", "tuberculosis", "hepatitis c", "cirrhosis", "diabetes",
    "food poisoning",
    "allergy",
    "chicken pox",
    "jaundice", "peptic ulcer diseae", "gastroenteritis",
    "bronchial asthma", "migraine", "cervical spondylosis"
]

for disease in emergency_diseases:
    if disease in disease_info:
        disease_info[disease]['emergency'] = True

mock_drug_data = {
    "paracetamol": {
        "name": "Paracetamol (Acetaminophen)",
        "description": "Commonly used for pain relief and fever reduction. Available over-the-counter.",
        "side_effects": ["Nausea", "Stomach pain", "Loss of appetite", "Dark urine", "Clay-colored stools", "Jaundice (rare)"],
        "dosage_info": "Adults: 500-1000 mg every 4-6 hours, max 4000 mg in 24 hours. Children: Dosage depends on weight, consult a doctor or pharmacist.",
        "link": "https://en.wikipedia.org/wiki/Paracetamol"
    },
    "ibuprofen": {
        "name": "Ibuprofen",
        "description": "A nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation.",
        "side_effects": ["Stomach upset", "Heartburn", "Nausea", "Vomiting", "Headache", "Dizziness", "Fluid retention"],
        "dosage_info": "Adults: 200-400 mg every 4-6 hours, max 1200 mg in 24 hours. Take with food or milk to reduce stomach upset.",
        "link": "https://en.wikipedia.org/wiki/Ibuprofen"
    },
    "decongestants": {
        "name": "Decongestants (e.g., Pseudoephedrine)",
        "description": "Used to relieve nasal congestion due to colds, allergies, and hay fever.",
        "side_effects": ["Nervousness", "Restlessness", "Difficulty sleeping", "Increased blood pressure", "Palpitations"],
        "dosage_info": "Follow label instructions. Not for prolonged use. Consult doctor for children or if you have heart conditions/high blood pressure.",
        "link": "https://en.wikipedia.org/wiki/Decongestant"
    },
    "cough syrup": {
        "name": "Cough Syrup (Various types)",
        "description": "Used to relieve cough symptoms. Can be suppressants (for dry cough) or expectorants (for productive cough).",
        "side_effects": ["Drowsiness", "Dizziness", "Nausea", "Vomiting"],
        "dosage_info": "Follow label instructions. Not all cough syrups are suitable for all ages or types of cough.",
        "link": "https://en.wikipedia.org/wiki/Cough_medicine"
    },
}

ambulance_contacts = {
    "India (General Emergency)": "108",
    "USA": "911",
    "UK": "999",
    "Canada": "911",
    "Australia": "000",
    "Germany": "112",
    "France": "112",
    "Japan": "119",
    "China": "120",
    "Brazil": "192",
    "India (Madhya Pradesh - MedCab)": "18008-908-208"
}

def normalize_symptom(symptom_text):
    cleaned_text = symptom_text.strip().lower()
    if not cleaned_text:
        return None
    corrected_text = str(TextBlob(cleaned_text).correct())
    if corrected_text.replace(' ', '_') in symptom_index:
        return corrected_text.replace(' ', '_')
    best_match = process.extractOne(corrected_text, valid_symptom_list, score_cutoff=85)
    if best_match:
        matched_symptom = best_match[0].replace(' ', '_')
        return matched_symptom
    return None

def get_prediction_details(disease_name):
    disease_lower = disease_name.lower()
    info = disease_info.get(disease_lower, {})
    return {
        'description': info.get('description', 'Description not available. Please consult a doctor.'),
        'precautions': info.get('precautions', []),
        'medications': info.get('medications', []),
        'diets': info.get('diets', []),
        'workouts': info.get('workouts', []),
        'emergency': info.get('emergency', False)
    }

# --- Flask Routes ---

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    prediction = None
    description = None
    precautions = []
    medications = []
    diets = []
    workouts = []
    emergency = False
    original_symptoms_input = ""
    recognized_symptoms = []

    current_region_ambulance = {
        "label": "India (General Emergency)",
        "number": ambulance_contacts["India (General Emergency)"]
    }

    if request.method == 'POST':
        original_symptoms_input = request.form.get('symptoms', '').strip()
        input_symptoms_raw = [s.strip() for s in original_symptoms_input.split(',') if s.strip()]

        if not input_symptoms_raw:
            flash('Please enter at least one symptom to get a prediction.', 'warning')
            return render_template('index.html', original_symptoms_input=original_symptoms_input, ambulance_contacts=ambulance_contacts, current_region_ambulance=current_region_ambulance)

        processed_symptoms = []
        for sym_raw in input_symptoms_raw:
            normalized_sym = normalize_symptom(sym_raw)
            if normalized_sym and normalized_sym not in processed_symptoms:
                processed_symptoms.append(normalized_sym)
                recognized_symptoms.append(sym_raw.title())

        generic_symptom_keywords = {
            'fever', 'high_fever', 'cough', 'cold', 'sore_throat', 'headache', 'fatigue',
            'body_ache', 'mild_fever', 'irritability', 'restlessness', 'congestion',
            'runny_nose', 'sneezing'
        }

        red_flag_symptoms = {
            'chest_pain', 'breathlessness', 'severe_abdominal_pain', 'vomiting',
            'diarrhoea', 'unexplained_weight_loss', 'night_sweats', 'blurred_vision',
            'slurred_speech', 'loss_of_balance', 'paralysis', 'dizziness',
            'sweating', 'indigestion', 'acidity', 'burning_micturition',
            'patches_in_throat', 'swelling_joints', 'stiff_neck', 'loss_of_smell'
        }

        has_generic = any(s in processed_symptoms for s in generic_symptom_keywords)
        has_red_flags = any(s in processed_symptoms for s in red_flag_symptoms)

        if 1 <= len(processed_symptoms) <= 4 and has_generic and not has_red_flags:
            prediction = "Common Viral Infection (e.g., Common Cold, Flu)"
            description = "These symptoms are frequently associated with common viral infections. Most cases resolve with rest and home care. If symptoms worsen or persist, please see a doctor."
            precautions = [
                "Get ample rest to aid recovery.",
                "Stay well-hydrated with water, clear broths, and juices.",
                "Avoid close contact with others to prevent spreading.",
                "Wash your hands frequently with soap and water.",
                "Consider over-the-counter medications for symptom relief (e.g., Paracetamol for fever, cough syrup for cough)."
            ]
            medications = ["Paracetamol", "Ibuprofen", "Decongestants", "Cough syrup"]
            diets = ["Warm clear liquids", "Soups", "Fruits rich in Vitamin C"]
            workouts = ["Avoid strenuous activity; focus on rest."]
            emergency = False

            if current_user.is_authenticated:
                current_user.add_prediction_to_history({
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symptoms': original_symptoms_input,
                    'prediction': prediction
                })
                db.session.commit()
            return render_template('index.html',
                                   prediction=prediction,
                                   description=description,
                                   precautions=precautions,
                                   medications=medications,
                                   diets=diets,
                                   workouts=workouts,
                                   emergency=emergency,
                                   original_symptoms_input=original_symptoms_input,
                                   recognized_symptoms=recognized_symptoms,
                                   ambulance_contacts=ambulance_contacts,
                                   current_region_ambulance=current_region_ambulance)
        if processed_symptoms:
            input_vector = np.zeros(len(X.columns))
            for sym in processed_symptoms:
                if sym in symptom_index:
                    input_vector[symptom_index[sym]] = 1

            input_vector = input_vector.reshape(1, -1)
            predicted_disease_label = model.predict(input_vector)[0]
            confidence = 1.0

            if len(processed_symptoms) < 2 or \
               (len(processed_symptoms) <= 3 and not has_red_flags):
                prediction = "Uncertain Diagnosis - Please Consult a Doctor"
                description = "Based on the limited or generic symptoms provided, the system cannot make a confident or accurate diagnosis. It is crucial to consult a healthcare professional for a proper assessment."
                precautions = ["Do not self-diagnose based on this information.", "Seek medical advice immediately for an accurate diagnosis.", "Provide all your symptoms to a doctor."]
                medications = []
                diets = []
                workouts = []
                emergency = True
            else:
                prediction = predicted_disease_label
                details = get_prediction_details(prediction)
                description = details['description']
                precautions = details['precautions']
                medications = details['medications']
                diets = details['diets']
                workouts = details['workouts']
                emergency = details['emergency']

            if current_user.is_authenticated:
                current_user.add_prediction_to_history({
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symptoms': original_symptoms_input,
                    'prediction': prediction
                })
                db.session.commit()
        else:
            flash('No recognizable symptoms were entered. Please try again with common symptom terms.', 'warning')

    return render_template('index.html',
                           prediction=prediction,
                           description=description,
                           precautions=precautions,
                           medications=medications,
                           diets=diets,
                           workouts=workouts,
                           emergency=emergency,
                           original_symptoms_input=original_symptoms_input,
                           recognized_symptoms=recognized_symptoms,
                           ambulance_contacts=ambulance_contacts,
                           current_region_ambulance=current_region_ambulance)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/history')
@login_required
def history():
    sorted_history = sorted(current_user.get_prediction_history(), key=lambda x: x['timestamp'], reverse=True)
    return render_template('history.html', history=sorted_history)

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/get_drug_details', methods=['POST'])
def get_drug_details():
    drug_name = request.json.get('drug_name', '').lower()
    details = mock_drug_data.get(drug_name)
    if details:
        return jsonify(details)
    else:
        return jsonify({
            "error": True,
            "name": drug_name.title(),
            "description": "Information not available for this drug in our database. Please consult a pharmacist or doctor.",
            "side_effects": [],
            "dosage_info": "Please consult a medical professional or pharmacist.",
            "link": None
        })

@app.route('/nearby_hospitals', methods=['POST'])
def nearby_hospitals():
    lat = request.json.get('lat')
    lon = request.json.get('lon')

    if lat is None or lon is None:
        return jsonify({"error": "Location (latitude and longitude) not provided."}), 400

    overpass_url = "http://overpass-api.de/api/interpreter"
    # Query to find nodes (points) with amenity=hospital within 5000m radius
    overpass_query = f"""
    [out:json];
    node(around:5000,{lat},{lon})["amenity"="hospital"];
    out body;
    """
    
    hospitals = []
    try:
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        for element in data.get('elements', []):
            if element.get('tags'):
                name = element['tags'].get('name', 'Unnamed Hospital')
                # Try to get full address or build from components
                address_parts = []
                if element['tags'].get('addr:housenumber'):
                    address_parts.append(element['tags']['addr:housenumber'])
                if element['tags'].get('addr:street'):
                    address_parts.append(element['tags']['addr:street'])
                if element['tags'].get('addr:full'): # OSM can have a full address tag
                    address_parts = [element['tags']['addr:full']]
                elif element['tags'].get('addr:place'):
                    address_parts.append(element['tags']['addr:place'])
                if element['tags'].get('addr:city'):
                    address_parts.append(element['tags']['addr:city'])
                if element['tags'].get('addr:postcode'):
                    address_parts.append(element['tags']['addr:postcode'])
                if element['tags'].get('addr:state'):
                    address_parts.append(element['tags']['addr:state'])
                
                address = ", ".join(address_parts) if address_parts else "Address not available"
                
                # Generate a simple OSM map link for display
                map_url = f"https://www.openstreetmap.org/?mlat={element['lat']}&mlon={element['lon']}#map=15/{element['lat']}/{element['lon']}"

                hospitals.append({
                    "name": name,
                    "address": address,
                    "lat": element['lat'],
                    "lon": element['lon'],
                    "map_url": map_url
                })
        
        if not hospitals:
            return jsonify({"message": "No hospitals found nearby via OpenStreetMap."}), 200

        return jsonify(hospitals)

    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        return jsonify({"error": f"Failed to fetch hospitals from OpenStreetMap API. Please try again later. ({e})"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode API response from OpenStreetMap."}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unknown error occurred while fetching hospitals. ({e})"}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)