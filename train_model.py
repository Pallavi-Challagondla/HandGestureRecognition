import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Folder with your .npy files
data_path = "gesture_data"
gestures = ["index_up", "open_palm",  "peace", "thumbs_up"]
 # replaced fist with index_up

all_data = []

for gesture in gestures:
    file = os.path.join(data_path, f"{gesture}.npy")
    gesture_data = np.load(file)  # shape: (num_samples, 64)
    all_data.append(gesture_data)

    # Generate mirrored left-hand data by flipping x-coordinates
    mirrored_data = gesture_data.copy()
    mirrored_data[:, ::3] = 1 - mirrored_data[:, ::3]  # flip x-axis
    all_data.append(mirrored_data)  # add left-hand version

# Combine all gestures (right + mirrored left)
dataset = np.vstack(all_data)
print("Dataset shape:", dataset.shape)  # double the samples now

# Split features and labels
X = dataset[:, :-1]  # 63 landmark values
y = dataset[:, -1]   # gesture labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)  # increased trees for better accuracy
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save the trained model
joblib.dump(model, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
