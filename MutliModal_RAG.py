# -------------------------------Imports------------------------------------
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
import torch
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
import fitz
from groq import Groq
import base64
import streamlit as st

# -------------------------------Groq API Setup------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
GROQ_API_KEY = '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'
client = Groq(api_key=GROQ_API_KEY)
llava_model = "llava-v1.5-7b-4096-preview"
llama_model = "llama3-groq-8b-8192-tool-use-preview"

# ---------------------------------Helper Functions--------------------------------
def encode_to_64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def image_to_text(client, model, base64_image, prompt):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ]
        }],
        model=model
    )
    return chat_completion.choices[0].message.content

def further_query(client, image_description, user_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an image description chatbot. Answer the user's queries based on the description."},
            {"role": "user", "content": f"{image_description}\n\n{user_prompt}"}
        ],
        model=llama_model
    )
    return chat_completion.choices[0].message.content

def complete_image_func(client, image_path, model, user_prompt):
    base64_image = encode_to_64(image_path)
    prompt = "Describe the image"
    image_description = image_to_text(client, model, base64_image, prompt)
    return further_query(client, image_description, user_prompt)

def extract_text_and_images_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        all_text += f"--- Page {page_num + 1} ---\n{page_text}\n"
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            img_filename = f"extracted_images/pdfs/image_page{page_num+1}_{img_index}.{image_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)
    return all_text

def scrape_page(url, web_dir):
    r = requests.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        for element in soup(['script', 'style']):
            element.extract()
        all_text = soup.get_text(separator=' ')
        clean_text = ' '.join(all_text.split())
        images = soup.find_all('img')
        image_urls = []
        os.makedirs(web_dir, exist_ok=True)
        for img in images:
            img_url = img.get('src')
            full_img_url = urljoin(url, img_url)
            if any(full_img_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_urls.append(full_img_url)
                try:
                    img_response = requests.get(full_img_url)
                    img_name = os.path.join(web_dir, os.path.basename(full_img_url))
                    with open(img_name, 'wb') as img_file:
                        img_file.write(img_response.content)
                except Exception as e:
                    print(f"Failed to save image from {full_img_url}: {e}")
        return clean_text, image_urls
    else:
        return None, None

def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if 'youtube.com' in parsed_url.netloc:
        if 'v=' in parsed_url.query:
            return parse_qs(parsed_url.query)['v'][0]
        path_segments = parsed_url.path.split('/')
        return path_segments[path_segments.index('watch') + 1] if 'watch' in path_segments else None
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    return None

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def chunk_content_by_sentence(text):
    return sent_tokenize(text)

def generate_rag_response(query, model, tokenizer, index, content_chunks):
    query_inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()
    k = 5
    _, indices = index.search(query_embedding, k)
    relevant_contexts = [content_chunks[i] for i in indices[0]]
    combined_context = " ".join(relevant_contexts)
    input_text = f"### Context Overview:\n{combined_context}\n\n### Question:\n{query}\n\n### Your Answer:"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": input_text}],
        model=llama_model,
        temperature=0.3
    ).choices[0].message.content
    return response.strip()

def final_func(user_query, user_image, pdf_file, url, youtube_url, model, tokenizer, web_dir="extracted_images/webscraping"):
    img_text = complete_image_func(client, user_image, llava_model, user_query)
    pdf_text = extract_text_and_images_from_pdf(pdf_file)
    web_text, _ = scrape_page(url, web_dir)
    video_id = extract_video_id(youtube_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    yt_text = " ".join([entry['text'] for entry in transcript])
    text = (web_text or "") + (pdf_text or "") + (yt_text or "") + (img_text or "")
    content_chunks = chunk_content_by_sentence(text)
    chunk_embeddings = []
    for chunk in content_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
        chunk_embeddings.append(embedding)
    embeddings_np = np.vstack(chunk_embeddings)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return generate_rag_response(user_query, model, tokenizer, index, content_chunks)

# -------------------------------Streamlit UI--------------------------------
user_db = {"test_user": "test_password"}

def validate_user(username, password):
    if username in user_db and user_db[username] == password:
        return True
    return False

def create_user(username, password):
    if username in user_db:
        st.error("Username already exists.")
    else:
        user_db[username] = password
        st.success(f"User '{username}' created successfully!")

import streamlit as st
import os
import pickle

# -------------------------------Helper Functions------------------------------------
USER_CREDENTIALS_FILE = "user_credentials.pkl"

def load_user_credentials():
    if os.path.exists(USER_CREDENTIALS_FILE):
        with open(USER_CREDENTIALS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_user_credentials(credentials):
    with open(USER_CREDENTIALS_FILE, 'wb') as f:
        pickle.dump(credentials, f)

def authenticate_user(username, password):
    credentials = load_user_credentials()
    if username in credentials and credentials[username] == password:
        return True
    return False

def register_user(username, password):
    credentials = load_user_credentials()
    if username in credentials:
        return False
    credentials[username] = password
    save_user_credentials(credentials)
    return True

# -------------------------------Login and Signup Pages------------------------------------
def login_page():
    st.title("Login to Medical AI Assistant")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.session_state.page = "main"
        else:
            st.error("Invalid username or password")

def signup_page():
    st.title("Signup for Medical AI Assistant")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    if st.button("Signup"):
        if password != confirm_password:
            st.error("Passwords do not match")
        elif register_user(username, password):
            st.success("Account created successfully. Please login.")
            st.session_state.page = "login"
        else:
            st.error("Username already exists. Please choose a different one.")


# -------------------------------Main Page (Doctor's AI Assistant)------------------------------------
def main_page():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        if "page" not in st.session_state or st.session_state.page == "login":
            page = st.sidebar.selectbox("Select a page", ["Login", "Signup"])
            if page == "Login":
                login_page()
            elif page == "Signup":
                signup_page()
        else:
            st.session_state.page = "login"
            login_page()
    else:
        st.title("Doctor's AI Assistant for Medical Services")
        st.image("medical_banner_1.jpg", use_container_width=True, caption="Empowering doctors with AI-driven assistance.")
        
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        webpage_url = st.sidebar.text_input("Enter Webpage URL")
        youtube_url = st.sidebar.text_input("Enter YouTube URL")
        if webpage_url == "":
            webpage_url = "https://www.webpagetest.org/blank.html"
        if youtube_url == "":
            youtube_url = "https://www.youtube.com/watch?v=1aA1WGON49E"
        saved_pdf_path = None
        saved_image_path = None

        if uploaded_file:
            saved_pdf_path = "./uploaded_document.pdf"
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if uploaded_image:
            saved_image_path = "./uploaded_image.jpg"
            with open(saved_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        user_query = st.chat_input("Ask a question about the medical case or procedure.")
        if user_query:
            response = final_func(user_query, saved_image_path, saved_pdf_path, webpage_url, youtube_url, model, tokenizer)
            st.session_state.chat_history.append({"question": user_query, "answer": response})

        for entry in st.session_state.chat_history:
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Response:** {entry['answer']}")

        with st.expander("Document Preview"):
            if uploaded_file and saved_pdf_path:
                with fitz.open(saved_pdf_path) as pdf:
                    for page_num, page in enumerate(pdf):
                        st.markdown(f"**Page {page_num + 1}**")
                        st.text(page.get_text("text")[:500])

        with st.expander("Image Description"):
            if uploaded_image and saved_image_path:
                img_text = complete_image_func(client, saved_image_path, llava_model, user_query)
                st.write(img_text)

        with st.expander("YouTube Transcript"):
            if youtube_url:
                video_id = extract_video_id(youtube_url)
                if video_id:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    yt_text = " ".join(entry['text'] for entry in transcript)
                    st.write(yt_text[:500])

if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    main_page()
else:
    main_page()
