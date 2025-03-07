import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set page config at the very beginning
#st.set_page_config(page_title="🌱 Karnataka Crop Recommendation", layout="centered")

# Load datasets
crop_dataset_path = "C:\\Users\\nitee\\Desktop\\LAST_FINAL DATASET.csv"
df = pd.read_csv(crop_dataset_path)

# Load additional datasets
yield_data = pd.read_csv("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\High yeild in varites.csv")
fertilizer_data = pd.read_csv("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\Fertilizer consumption.csv")
price_data = pd.read_csv("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\price.csv")

# Ensure columns are clean
df["Location"] = df["Location"].astype(str).str.strip()
df = df.dropna(subset=["Location", "Season", "Soil type", "Crops"])

# Encode categorical values for training
le_location = LabelEncoder()
le_season = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

df["Location"] = le_location.fit_transform(df["Location"])
df["Season"] = le_season.fit_transform(df["Season"])
df["Soil type"] = le_soil.fit_transform(df["Soil type"])
df["Crops"] = le_crop.fit_transform(df["Crops"])

# Train a Random Forest model
X = df[["Location", "Season", "Soil type"]]
y = df["Crops"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "crop_recommendation_model.pkl")

# Streamlit UI
def main():
    st.title("🚜 Karnataka Crop Recommendation System")
    st.write("### Select your district and season to get the best crop recommendations!")

    # User input
    district = st.selectbox("🏙 Select District", le_location.classes_)
    season = st.selectbox("⏳ Select Season", le_season.classes_)
    filtered_soil_types = df[df["Location"] == le_location.transform([district])[0]]["Soil type"].unique()

    if len(filtered_soil_types) > 0:
        soil_type = st.selectbox("🌍 Select Soil Type", le_soil.inverse_transform(filtered_soil_types))
    else:
        st.warning("⚠ No soil data available for this district and season.")
        return

    # Predict crops using the trained model
    if st.button("🌾 Recommend Crops"):
        model = joblib.load("crop_recommendation_model.pkl")
        input_data = [[le_location.transform([district])[0], le_season.transform([season])[0], le_soil.transform([soil_type])[0]]]
        predicted_crop = le_crop.inverse_transform(model.predict(input_data))
        st.success(f"✅ Recommended Crop: {predicted_crop[0]}")

    # Visualization button
    if st.button("📊 Visualize Data"):
        plot_visualizations(district)

# Function to display visualizations
def plot_visualizations(district):
    st.write(f"### Data Visualization for {district}")
    
    district_data = yield_data[yield_data["Dist Name"] == district]

    if district_data.empty:
        st.warning("⚠ Data not available for this district.")
        return

    fig, ax = plt.subplots()
    sns.lineplot(data=district_data, x="Year", y="TOTAL AREA (1000 ha)", marker="o", ax=ax)
    ax.set_title(f"Yield Trends in {district}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (kg/ha)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
