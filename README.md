from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = 'resumes'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text(file):
    filename = file.filename
    if filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith('.docx'):
        return docx2txt.process(file)
    return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else ""

def extract_phone(text):
    match = re.search(r'(\+\d{1,3}\s?)?(\d{10})', text)
    return match.group(0) if match else ""

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        job_description = request.form['jobdesc']
        resume_file = request.files['resume']
        resume_text = extract_text(resume_file)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        score = calculate_similarity(job_description, resume_text)
        result = {
            'email': email,
            'phone': phone,
            'score': round(score * 100, 2)
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
