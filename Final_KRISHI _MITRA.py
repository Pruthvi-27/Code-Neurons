# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 07:14:20 2025

@author: nitee
"""

import streamlit as st
import pandas as pd
import hashlib
import os
import base64
import streamlit as st

st.set_page_config(
    page_title="Karnataka Agricultural AI System",
    page_icon="🌾"
)


# File to store user credentials
USER_DATA_FILE = "user_data.csv"

# Ensure user data file exists
if not os.path.exists(USER_DATA_FILE):
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_DATA_FILE, index=False)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check user credentials
def authenticate_user(username, password):
    users = pd.read_csv(USER_DATA_FILE)
    if username in users["username"].values:
        stored_password = users.loc[users["username"] == username, "password"].values[0]
        return stored_password == hash_password(password)
    return False

# Function to register a new user
def register_user(username, password):
    users = pd.read_csv(USER_DATA_FILE)
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    new_user.to_csv(USER_DATA_FILE, mode="a", header=False, index=False)
    return True

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Session state for tracking login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

def set_page(page):
    st.session_state.page = page
    st.rerun()

# ===================== LOGIN & SIGNUP PAGE =====================
if st.session_state.page == "login":
    set_background("C:\\Users\\nitee\\Desktop\\log.jpg")
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>🔒 Welcome to Karnataka Agricultural AI System</h1>", unsafe_allow_html=True)
    option = st.radio("**Choose an option:**", ["Login", "Sign in"])
    
    if option == "Login":
        username = st.text_input("**👤 Username**")
        password = st.text_input("**🔑 Password**", type="password")
        if st.button("**Login**"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                set_page("main")
            else:
                st.error("Invalid username or password!")
    elif option == "Sign in":
        new_username = st.text_input("**👤 Choose a Username**")
        new_password = st.text_input("**🔑 Choose a Password**", type="password")
        if st.button("**Sign in**"):
            if register_user(new_username, new_password):
                st.success("✅ Account created! Please log in.")
            else:
                st.error("❌ Username already exists. Try another.")

# ===================== MAIN MENU =====================
elif st.session_state.page == "main" and st.session_state.logged_in:
    set_background("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\login_image.jpg")
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>🌾 Karnataka Agricultural AI System</h1>", unsafe_allow_html=True)
    st.write("### **Select an option below:**")
    if st.button("🌾 Crop Recommendation"):
        set_page("crop_recommendation")
    if st.button("🌱 Fertilizer Recommendation"):
        set_page("fertilizer_recommendation")
    if st.button("🦠 Crop Disease Detection"):
        set_page("plant_disease_detection")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        set_page("login")
        
elif st.session_state.page == "crop_recommendation":
    set_background("C:\\Users\\nitee\\Desktop\\croprec.jpg")
    exec(open("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\Crop_Recomendation.py", encoding="utf-8").read())
    if st.button("🔙 Back"):
        set_page("main")

# ===================== FERTILIZER RECOMMENDATION =====================
elif st.session_state.page == "fertilizer_recommendation":
    set_background("C:\\Users\\nitee\\Desktop\\leaff.jpg")
    exec(open("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\complete_fertile.py", encoding="utf-8").read())
    if st.button("🔙 Back"):
        set_page("main")

# ===================== PLANT DISEASE DETECTION =====================
elif st.session_state.page == "plant_disease_detection":
    set_background("C:\\Users\\nitee\\Desktop\\predicit.jpg")
    exec(open("C:\\Users\\nitee\\Desktop\\JSSDATASET\\Prediction\\Recomendation Frontend.py", encoding="utf-8").read())
    if st.button("🔙 Back"):
        set_page("main")