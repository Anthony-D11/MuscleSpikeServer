import os
import shutil
from typing import List
from fastapi import FastAPI, File, Response, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import m2cgen as m2c

import libemg
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGClassifier

# Constants
MYO_FS = 200    # MyoBand sampling frequency (Hz)
DATA_FOLDER = 'data/'

app = FastAPI(title="MuscleSpike Server")
os.makedirs(DATA_FOLDER, exist_ok=True)

def get_window_parameters(fs):
    window_size_seconds = 0.25
    window_increment_seconds = 0.05
    return int(fs * window_size_seconds), int(fs * window_increment_seconds)

def extract_features_labels(odh):
    window_size, window_increment = get_window_parameters(MYO_FS)
    windows, metadata = odh.parse_windows(window_size, window_increment)

    fe = FeatureExtractor()
    features = fe.extract_features(['MAV', 'ZC', 'SSC', 'WL'], windows, array=True)
    return features, metadata['classes']

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint for the mobile app to send CSV files.
    Files should be named following the C_0_R_0_emg.csv convention.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_files = []
    
    for uploaded_file in files:
        if uploaded_file.filename == '':
            continue
        
        filepath = os.path.join(DATA_FOLDER, uploaded_file.filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
            
        saved_files.append(uploaded_file.filename)

    return {"message": f"Successfully saved {len(saved_files)} files.", "files": saved_files}

@app.get("/train")
def train_and_minify():
    """
    Endpoint to trigger training on the current data folder, 
    compile the minified model, and return it to the app as JSON.
    """
    if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
        raise HTTPException(status_code=400, detail="No data found. Upload files first.")

    print("Loading recorded trials...")
    offline_odh = OfflineDataHandler()
    offline_odh.get_data(folder_location=DATA_FOLDER,
                         regex_filters=[
                             RegexFilter("data/C_", "_R_", ["0","1","2","3","4"], "classes"),
                             RegexFilter("_R_", "_emg.csv", ["0","1","2","3","4"], "reps")
                         ])

    print("Extracting features and formatting labels...")
    features, raw_labels = extract_features_labels(offline_odh)
    
    gesture_map = {
        "0": "No Movement",
        "1": "Hand Open",
        "2": "Hand Close",
        "3": "Wrist Extension",
        "4": "Wrist Flexion"
    }

    named_labels = [gesture_map[str(label)] for label in raw_labels]

    print("Training RF Classifier...")
    clf = EMGClassifier('RF')
    clf.fit({
        'training_features': features,
        'training_labels': raw_labels
    })

    # -------------------------------------------------------------------------
    # "Minify" the Model for Mobile
    # -------------------------------------------------------------------------
    # sklearn_model = clf.predictor.model
    
    # minified_model = {
    #     "architecture": "LDA",
    #     "features_required": ["MAV", "ZC", "SSC", "WL"],
    #     "classes": sklearn_model.classes_.tolist(),
    #     "weights": sklearn_model.coef_.tolist(),
    #     "bias": sklearn_model.intercept_.tolist()
    # }

    print("Model successfully minified! Sending payload to mobile app.")
    
    # Optional: Clean up after successful training
    # shutil.rmtree(DATA_FOLDER)
    # os.makedirs(DATA_FOLDER)

    model_in_js = m2c.export_to_javascript(clf.model)
    wrapped_code = f"{model_in_js}\nreturn score(input);"
    print(wrapped_code)
    return {
        "status": "success",
        "message": "Model compiled successfully.",
        "model": wrapped_code
    }

@app.get("/ping")
def verify_connection():
    """
    Simple health check endpoint for the mobile app to verify 
    that the server is online and reachable.
    """
    return {
        "status": "success", 
        "message": "Connection established! The MuscleSpike server is running."
    }

if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)