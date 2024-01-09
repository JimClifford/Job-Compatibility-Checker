import os
import openai
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import io
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set OpenAI API key
openai.api_key = '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'

st.title("Job Compatibility Checker.")
st.write("You have the flexibility to upload and manage multiple resumes and job descriptions.")
st.write("Allowing you the freedom to select and present your most relevant and compelling documents for the job.")

# Function to open a file dialog and return the selected file path
def get_resume_paths():
    resume_files = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    resume_paths = []

    for resume_file in resume_files:
        # Save the file content to a temporary file
        resume_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        resume_tempfile.write(resume_file.read())
        resume_paths.append((resume_tempfile.name, resume_file.name))

    return resume_paths

#Function to select the job desription
def get_job_description():
    job_file = st.file_uploader("Upload the corresponding job description", type=["pdf"])
    if job_file is not None:
        # Save the file content to a temporary file
        job_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        job_tempfile.write(job_file.read())
        return job_tempfile.name
    return None

def generate_pdf_output(responses, num_resumes_to_display):
    pdf_filename = "output_report.pdf"

    # Create a PDF document
    pdf_canvas = canvas.Canvas(pdf_filename, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)

    # Add content to the PDF
    pdf_canvas.drawString(100, 750, f"Top {num_resumes_to_display} Resumes Based on Similarity:")
    y_position = 730

    for i, (resume_path, similarity) in enumerate(responses[:num_resumes_to_display]):
        resume_name = os.path.basename(resume_path)
        pdf_canvas.drawString(100, y_position, f"Resume {i + 1}: {resume_name} - Similarity: {similarity}")
        y_position -= 20

    pdf_canvas.save()

    st.success(f"PDF report generated: {pdf_filename}")

    # Add a download link
    st.download_button(
        label="Download PDF Report",
        data=open(pdf_filename, "rb").read(),
        key="download_button",
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Get the PDF file paths from the user
pdf_Rpaths = get_resume_paths()
pdf_JDpath = get_job_description()

if pdf_JDpath is not None and pdf_Rpaths:
    loader_JD = PyMuPDFLoader(pdf_JDpath)
    documents_JD = loader_JD.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    textsJD = text_splitter.split_documents(documents_JD)

    responses = []

    for pdf_Rpath, resume_name in pdf_Rpaths:
        loader_R = PyMuPDFLoader(pdf_Rpath)
        documents_R = loader_R.load()

        textsR = text_splitter.split_documents(documents_R)
        togetherTexts = textsR + textsJD

        def get_completion(prompt, model="gpt-4-1106-preview"):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,  # this is the degree of randomness of the model's output
            )
            return response.choices[0].message["content"]

        prompt = f"""
        Perform the following actions:
        1. Identify the key skills, qualifications, and experience required in the job description file.
        2. Compare these skills, qualifications, and experiences with the ones in the resume file.
        3. Give a percentage for similarity, indicating how well the user's resume aligns with the job requirements.

        Review text: '''{togetherTexts}'''
        """
        response = get_completion(prompt)
        responses.append((resume_name, response))

    # Sort responses based on similarity scores
    responses.sort(key=lambda x: float(x[1].split(' ')[-1].replace('%', '')) if x[1].replace('%', '').isdigit() else 0, reverse=True)

    # Display the selected number of top resumes
    num_resumes_to_display = st.number_input("Please select the number for the best resumes you want to display.", min_value=1, max_value=len(responses), value=1)

    st.header(f"Top {num_resumes_to_display} Resumes:")
    for i, (resume_name, similarity) in enumerate(responses[:num_resumes_to_display]):
        st.text(f"Resume {i + 1}: {resume_name} - Similarity: {similarity}")

    # Generate PDF report
    generate_pdf_output(responses, num_resumes_to_display)

else:
    st.warning("Please upload the job description and at least one resume as PDF files")

