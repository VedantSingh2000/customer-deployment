import streamlit as st

st.title("Hello Streamlit!")

st.write("This is a very basic Streamlit app.")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello, {name}!")

if st.button("Click me"):
    st.write("You clicked the button!")
