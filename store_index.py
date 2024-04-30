import streamlit as st
import os
import tempfile

# Title
st.title("File Uploader")

# Create a temporary directory
temp_dir = 'Data/temp'

# File uploader
uploaded_file = st.file_uploader("Choose a file")

# Display file details if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display file details
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    # Display file contents
    st.write("File content:")
    with open(file_path, "r") as f:
        file_contents = f.read()
        st.code(file_contents)
