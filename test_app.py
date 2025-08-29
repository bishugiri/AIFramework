import streamlit as st
import os

# Test dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    st.write("dotenv loaded successfully")
except Exception as e:
    st.write(f"dotenv error: {e}")

st.title("Test App")
st.write("If you see this, the app is working!")
