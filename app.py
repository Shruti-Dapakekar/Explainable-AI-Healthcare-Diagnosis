from flask import Flask, render_template, request
import pickle
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # Prevents GUI issues in Flask # Run matplotlib without opening a window
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("models/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
X_train = pickle.load(open("models/X_train.pkl", "rb"))

explainer = shap.Explainer(model, X_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Pregnancies": float(request.form["Pregnancies"]),
        "Glucose": float(request.form["Glucose"]),
        "BloodPressure": float(request.form["BloodPressure"]),
        "SkinThickness": float(request.form["SkinThickness"]),
        "Insulin": float(request.form["Insulin"]),
        "BMI": float(request.form["BMI"]),
        "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
        "Age": float(request.form["Age"])
    }

    input_df = pd.DataFrame([data])

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)
    shap_values = explainer(scaled) #generate SHAP values for the input data
# Create a new figure for SHAP explanation
    plt.figure(figsize=(8,5))

# Draw SHAP waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)

# Save plot so Flask can display it
    plt.savefig("static/shap/shap_plot.png", bbox_inches="tight")

# Check whether the image was created successfully
    print("Image Exists:", os.path.exists("static/shap/shap_plot.png"))
# Close figure to free memory
    plt.close()
    
    probability = model.predict_proba(scaled)
    confidence = max(probability[0]) * 100

# Create a new figure (canvas) where the SHAP explanation graph will be drawn.
   # plt.figure()

# Generate a waterfall plot for the first prediction.
# shap_values[0] means explain the first (and only) patient prediction.
# show=False prevents the graph from opening in a separate window.
# This is important because Flask runs on a server, not on your desktop.
    #shap.plots.waterfall(shap_values[0], show=False)

# Save the generated SHAP graph as an image.
# It will be stored inside:
# static/
#    └── shap/
#         └── shap_plot.png
#
# Flask can display any file inside the "static" folder.
   # plt.savefig("static/shap/shap_plot.png", bbox_inches="tight")

# Close the figure to free memory.
# Without this, every prediction creates another figure and memory usage grows.
   # plt.close()


    if prediction[0] == 1:
        result = "Patient is Diabetic"
    else:
        result = "Patient is Not Diabetic"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        shap_plot="static/shap/shap_plot.png"  # Send SHAP image to HTML
    )

if __name__ == "__main__":
    app.run(debug=True)