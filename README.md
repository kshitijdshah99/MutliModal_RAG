# Multimodal RAG (Retrieval-Augmented Generation) 🚀

Welcome to the **Multimodal RAG** project! This repository provides an end-to-end implementation of a multimodal retrieval-augmented generation system. It can extract, process, and query information from diverse sources, including PDFs, images, web pages, and YouTube videos, while leveraging advanced AI models to deliver intelligent, context-aware responses. 💡

---

## 🌟 Key Features

- **Multimodal Input Support** 🖼️📄📹:
  - **PDFs**: Extract text and images, ready for efficient retrieval.
  - **Images**: Convert visuals into descriptive text with Groq’s LLaVA model.
  - **Web Pages**: Scrape and process textual and visual content from URLs.
  - **YouTube Videos**: Retrieve and process video transcripts seamlessly.

- **AI-Powered Query Response** 🤖:
  - Combines extracted data into meaningful chunks.
  - Delivers context-aware answers using Groq’s LLaMA model.

- **Streamlit UI** 🖥️:
  - Interactive and user-friendly interface for file uploads and query submissions.
  - Preview documents, describe images, and access YouTube transcripts directly.

- **Authentication System** 🔒:
  - Secure login and signup functionality to protect user data.

---

## 🛠️ Technologies Used

- **Programming Language**: Python 🐍
- **Frameworks and Libraries**:
  - [Streamlit](https://streamlit.io) 🌐: For building the interactive UI.
  - [Transformers](https://huggingface.co/transformers) 🤗: For tokenization and AI model integration.
  - [FAISS](https://faiss.ai) 🔍: For efficient similarity search and indexing.
  - [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) 📄: To extract text and images from PDFs.
  - [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) 🎥: For fetching video transcripts.
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) 🥣: For web scraping.
- **Models**:
  - Groq’s **LLaVA v1.5**: A multimodal language-vision model for image understanding.
  - Groq’s **LLaMA v3**: Advanced language model for RAG.
  - SentenceTransformers’ **all-mpnet-base-v2**: For generating embeddings.

---

## 🔍 How It Works

### 1. Data Ingestion 📥

#### PDFs
- Extracts text and images from uploaded PDF files.
- Stores images locally for further processing.

#### Images
- Converts uploaded images into base64 format.
- Uses Groq’s LLaVA model to generate descriptive text.

#### Web Pages
- Scrapes text and downloads images from specified URLs.
- Cleans and processes the scraped content.

#### YouTube Videos
- Extracts transcripts using the YouTube Transcript API.

### 2. Content Processing ⚙️
- Tokenizes text into sentences for better indexing.
- Embeds the content using SentenceTransformers.
- Indexes embeddings using FAISS for efficient retrieval.

### 3. Query Resolution 💬
- Accepts user queries via Streamlit’s chat interface.
- Searches the indexed content for the most relevant context.
- Generates context-aware responses using Groq’s LLaMA model.

---

## 🚀 Getting Started

### Prerequisites ✅

- Python 3.9+
- API key for Groq (required for LLaVA and LLaMA models)

### Installation 🛠️

1. Clone this repository:
   ```bash
   git clone https://github.com/kshitijdshah99/MultiModal_RAG.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Groq API:
   - Replace `GROQ_API_KEY` in the code with your API key.

4. Run the Streamlit app:
   ```bash
   streamlit run MultiModal_RAG.py
   ```

---

## 🧑‍💻 Usage

1. **Login or Signup** 🔐:
   - Use the Streamlit interface to create an account or log in.

2. **Upload Files** 📤:
   - Upload PDFs or images using the sidebar.

3. **Enter URLs** 🌐:
   - Provide web page URLs or YouTube video links for processing.

4. **Ask Questions** 🤔:
   - Submit queries via the chat interface.

5. **View Results** 👀:
   - Preview document text, image descriptions, and YouTube transcripts.

---

## 📂 File Structure

- `app.py`: Main Streamlit application.
- `requirements.txt`: List of dependencies.
- `user_credentials.pkl`: File for storing user credentials.
- `extracted_images/`: Directory for storing extracted images.

---

## 🚧 Future Enhancements

- Support for additional file formats (e.g., DOCX, XLSX).
- Improved indexing and retrieval algorithms.
- Enhanced visualization of retrieved content.

---

## 🤝 Contributions

Contributions are welcome! Please submit issues or pull requests to help improve this project.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---


