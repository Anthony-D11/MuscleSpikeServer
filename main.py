import libemg
from libemg.data_handler import OfflineDataHandler, OnlineDataHandler, RegexFilter
from libemg.streamers import myo_streamer
from libemg.gui import GUI
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier 
import time

# Constants
MYO_FS = 200    # MyoBand sampling frequency (Hz)

def get_window_parameters(fs):
    window_size_seconds = 0.25
    window_increment_seconds = 0.05
    return int(fs * window_size_seconds), int(fs * window_increment_seconds)

def extract_features_labels(odh):
    # Simplified for real-time classification parity. 
    # (Filters removed here to ensure raw training data matches raw live streaming data)
    window_size, window_increment = get_window_parameters(MYO_FS)
    windows, metadata = odh.parse_windows(window_size, window_increment)

    fe = FeatureExtractor()
    features = fe.extract_feature_group('HTD', windows, array=True)
    return features, metadata['classes']

def main():
    # -------------------------------------------------------------------------
    # 1. Start the Hardware Stream
    # -------------------------------------------------------------------------
    print("Starting Myo Streamer...")
    myo_process, shared_memory_items = myo_streamer()
    online_data_handler = OnlineDataHandler(shared_memory_items=shared_memory_items)
    
    # -------------------------------------------------------------------------
    # 2. Record Your Personal Training Data
    # -------------------------------------------------------------------------
    print("Launching GUI. Please record your gestures.")
    gui = GUI(online_data_handler=online_data_handler)
    gui.download_gestures(gesture_ids=[1,2,3,4,5], folder='images/')
    
    # This blocks execution until you finish recording and close the GUI window.
    # CRITICAL: Configure the GUI to save your trials to a folder named "data/"
    gui.start_gui() 

    # -------------------------------------------------------------------------
    # 3. Load the Recorded Data
    # -------------------------------------------------------------------------
    print("Loading recorded trials...")
    offline_odh = OfflineDataHandler()
    offline_odh.get_data(folder_location="data/",
                         regex_filters=[
                             RegexFilter("data/C_", "_R_", ["0","1","2","3","4"], "classes"),
                             RegexFilter("_R_", "_emg.csv", ["0","1","2","3","4"], "reps")
                         ])

    # -------------------------------------------------------------------------
    # 4. Extract Features and Train the Model
    # -------------------------------------------------------------------------
    print("Training model...")
    
    features, labels = extract_features_labels(offline_odh)
    
    gesture_map = {
        "0": "No Movement",
        "1": "Hand Open",
        "2": "Hand Close",
        "3": "Wrist Extension",
        "4": "Wrist Flexion"
    }

    # 2. Swap out the numbers for the actual names
    # Note: We ensure the raw_label is cast to a string just in case LibEMG parsed it as an int!
    named_labels = [gesture_map[str(label)] for label in labels]

    clf = EMGClassifier('RF')
    clf.fit({
        'training_features': features,
        'training_labels': labels
    })
    print("Model trained successfully!")

    # -------------------------------------------------------------------------
    # 5. Run Live Real-Time Predictions
    # -------------------------------------------------------------------------
    print("Starting live predictions... Move your arm to see outputs!")
    window_size, window_increment = get_window_parameters(MYO_FS)
    
    online_classifier = OnlineEMGClassifier(
        offline_classifier=clf,
        window_size=window_size,
        window_increment=window_increment,
        online_data_handler=online_data_handler,
        output_format='probabilities',
        features=['MAV', 'ZC', 'SSC', 'WL'], # Tells the online classifier to extract the same feature group
        std_out=True      # Prints the predictions directly to your VS Code terminal
    )
    
    # This keeps the script running endlessly to process the live feed
    online_classifier.run(block=True) 

if __name__ == '__main__':     
    main()