from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import os

#Load Dataset

X, y = load_iris(return_X_y=True)

#Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Create folder if does not exists

os.makedirs("model", exist_ok=True)

# Save the model

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
    
print("model.pkl created successfully!")
