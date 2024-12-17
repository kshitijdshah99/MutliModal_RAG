# Multimodal RAG (Retrieval-Augmented Generation) 🚀🚀

Welcome to the **Multimodal RAG** project! This repository is a powerhouse of innovation, enabling seamless integration of diverse data sources into one unified, intelligent system. Whether it's scraping text from PDFs, extracting insights from images, processing web content, or deciphering YouTube transcripts, **Multimodal RAG** is here to revolutionize your data retrieval and question-answering workflows. 🎉

---

## 🌟 Key Features

### Multimodal Input Support 🖼️📄📹
- **PDFs**: Extract both text and images for streamlined analysis.
- **Images**: Leverage Groq’s **LLaVA** model to generate rich descriptive captions.
- **Web Pages**: Scrape and organize text and images from URLs with precision.
- **YouTube Videos**: Fetch and process transcripts to make video content searchable.

### AI-Powered Query Resolution 🤖💬
- Combine multiple data sources into meaningful, context-rich answers.
- Powered by Groq’s **LLaMA model**, delivering intelligent and accurate responses.

### User-Friendly Streamlit Interface 🖥️🎨
- An intuitive UI to upload files, submit queries, and interact with processed content.
- Preview documents, describe images, and retrieve YouTube transcripts effortlessly.

### Secure User Authentication 🔒🛡️
- Robust login and signup system ensures data security and personalized access.

---

## 🛠️ Technologies Used

- **Programming Language**: Python 🐍
- **Frameworks and Libraries**:
  - [Streamlit](https://streamlit.io) 🌐: For a sleek and interactive UI.
  - [Transformers](https://huggingface.co/transformers) 🤗: Tokenization and model integration.
  - [FAISS](https://faiss.ai) 🔍: Fast similarity search and indexing.
  - [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) 📄: PDF text and image extraction.
  - [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) 🎥: For seamless video transcript fetching.
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) 🥣: Efficient web scraping.

- **Models**:
  - **Groq’s LLaVA v1.5**: A cutting-edge multimodal vision-language model for image understanding.
  - **Groq’s LLaMA v3**: Advanced language model tailored for RAG applications.
  - **SentenceTransformers all-mpnet-base-v2**: Generates high-quality embeddings for indexing and search.

---

## 🔍 How It Works

### 1. Data Ingestion 📥

#### PDFs 📄
- Extracts text and images from uploaded PDF files.
- Stores extracted images locally for further processing and retrieval.

#### Images 🖼️
- Converts images to base64 format for analysis.
- Uses **LLaVA** to create descriptive captions for better understanding.

#### Web Pages 🌐
- Scrapes text and downloads images from specified URLs.
- Cleans and preprocesses the content for effective querying.

#### YouTube Videos 🎥
- Retrieves transcripts using the **YouTube Transcript API**.

### 2. Content Processing ⚙️
- Tokenizes extracted content into manageable chunks.
- Embeds these chunks using **SentenceTransformers** for semantic similarity.
- Indexes embeddings with **FAISS**, enabling lightning-fast retrieval.

### 3. Query Resolution 💬🤔
- Accepts natural language queries through a chat-like interface.
- Identifies relevant content chunks from indexed data.
- Generates accurate, context-rich answers using **LLaMA**.

---

## 🚀 Getting Started

### Prerequisites ✅
- Python 3.9+
- API key for Groq (required for using **LLaVA** and **LLaMA** models)

### Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/kshitijdshah99/MultiModal_RAG.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the **Groq API**:
   - Replace `GROQ_API_KEY` in the code with your API key.

4. Run the Streamlit app:
   ```bash
   streamlit run MultiModal_RAG.py
   ```

---

## 🧑‍💻 Usage

1. **Login or Signup** 🔐:
   - Create an account or log in through the Streamlit interface.

2. **Upload Files** 📤:
   - Upload PDFs or images using the intuitive sidebar.

3. **Enter URLs** 🌐:
   - Add web page URLs or YouTube video links to analyze their content.

4. **Ask Questions** 🤔:
   - Type your queries into the chat input field and get instant, context-aware responses.

5. **View Results** 👀:
   - Use the provided preview sections to examine document text, image captions, or video transcripts.

---

## User Interface👀
Login/Sign-up Page
![](https://github.com/kshitijdshah99/MutliModal_RAG/blob/main/Login_page.png)

Home Page
![](https://github.com/kshitijdshah99/MutliModal_RAG/blob/main/Frontend.png)
 

## 📂 File Structure

- `MultiModal_RAG.py`: Main Streamlit application script.
- `requirements.txt`: List of Python dependencies.
- `user_credentials.pkl`: Encrypted file storing user authentication details.
- `extracted_images/`: Directory for images extracted from PDFs or scraped from web pages.

---

## 🚧 Future Enhancements

- Support for more file types (e.g., DOCX, XLSX, Markdown).
- Enhanced user interface with drag-and-drop functionality.
- Improved context merging and answer generation.
- Integration of audio transcription and analysis.

---

## 🤝 Contributions

We’d love to hear from you! Whether it’s bug fixes, feature suggestions, or new ideas, contributions are always welcome. Feel free to fork the repository, make your changes, and submit a pull request. ✨

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
