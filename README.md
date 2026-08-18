# AI-Based Plagiarism Detection & Essay Evaluation System

An AI-assisted academic writing analysis system developed as a **collaborative final-year project**. The application combines web crawling, text processing, plagiarism detection, AI-content comparison, and automated essay evaluation to provide a comprehensive analysis of submitted essays.

---

## 🚀 Overview

The system allows users to submit an essay along with its topic and analyzes the content for potential plagiarism and writing quality.

The application performs the following major tasks:

* Searches the web for relevant documents related to the essay topic.
* Crawls and extracts textual content from relevant web pages.
* Preprocesses the retrieved content and submitted essay.
* Detects textual similarity using **TF-IDF vectorization and cosine similarity**.
* Generates AI-written essays for comparison with the submitted content.
* Calculates similarity between the submitted essay and AI-generated reference content.
* Evaluates the quality of the essay using the **Google Gemini API**.
* Generates plagiarism results and detailed essay feedback.

The application is implemented as an interactive **Streamlit** web application.

---

## ✨ Features

### 🔍 Plagiarism Detection

The system analyzes the submitted essay against relevant online sources to identify potential textual overlap.

Key features include:

* Web-based source retrieval.
* Text extraction from web pages.
* Text preprocessing.
* TF-IDF vectorization.
* Cosine similarity calculation.
* Similarity scoring between the submitted essay and retrieved sources.
* Ranking of sources based on similarity.
* Storage of plagiarism analysis results.

### 🌐 Web Crawling

The system retrieves relevant online content that can be used as reference material during plagiarism analysis.

The web crawling pipeline:

1. Receives the essay topic.
2. Searches for relevant online sources.
3. Retrieves web pages.
4. Extracts useful textual content.
5. Processes the extracted text.
6. Passes the processed content to the plagiarism detection pipeline.

This allows the system to compare the submitted essay against dynamically retrieved web content.

### 🤖 AI-Content Analysis

The system generates AI-written reference essays for a given topic and compares them with the submitted essay.

The analysis provides an AI-content similarity score based on textual similarity with generated reference essays.

> **Note:** AI-content analysis is similarity-based and should not be considered definitive proof that a particular essay was written by AI.

### 📝 Automated Essay Evaluation

The system uses the **Google Gemini API** to evaluate submitted essays across multiple criteria.

The evaluation includes:

* Coherence & Organization
* Grammar & Syntax
* Relevance to the Topic
* Use of Evidence & Examples
* Vocabulary & Language Variety
* Critical Thinking & Argument Strength

The system provides criterion-wise scores along with explanations, strengths, and areas for improvement.

### 📊 Results & Reporting

The system generates and stores analysis results, including:

* Plagiarism similarity scores
* Retrieved source information
* AI-generated reference essays
* AI-content similarity results
* Essay evaluation results
* CSV-based analysis data

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         │ Topic + Essay Text  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Interface │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
      │ Web Crawling  │     │ AI Content    │     │ Essay         │
      │ & Retrieval   │     │ Comparison    │     │ Evaluation    │
      └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
      │ Text          │     │ AI-Generated  │     │ Gemini API    │
      │ Extraction    │     │ Reference     │     │ Evaluation    │
      └───────┬───────┘     │ Essays        │     └───────┬───────┘
              │              └───────┬───────┘             │
              ▼                      ▼                     ▼
      ┌────────────────┐     ┌───────────────┐     ┌───────────────┐
      │ TF-IDF         │     │ Cosine        │     │ Criterion-wise│
      │ Vectorization  │     │ Similarity    │     │ Scoring       │
      └────────┬───────┘     └───────┬───────┘     └───────┬───────┘
               │                     │                     │
               └─────────────────────┼─────────────────────┘
                                     │
                                     ▼
                         ┌─────────────────────┐
                         │   Analysis Report   │
                         │                     │
                         │ • Similarity Scores │
                         │ • AI Similarity     │
                         │ • Essay Evaluation  │
                         │ • Feedback          │
                         └─────────────────────┘
```

---

## 🧠 Plagiarism Detection Pipeline

The plagiarism detection component uses **TF-IDF vectorization and cosine similarity** to measure textual similarity between the submitted essay and retrieved source documents.

```text
Submitted Essay
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Numerical Text Representation
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
Submitted Essay Vector        Source Document Vectors
       │                              │
       └──────────────┬───────────────┘
                      ▼
             Cosine Similarity
                      │
                      ▼
              Similarity Scores
                      │
                      ▼
             Ranked Web Sources
```

### TF-IDF

TF-IDF represents the importance of words within the submitted essay and reference documents.

It helps transform textual documents into numerical vectors that can be compared mathematically.

### Cosine Similarity

Cosine similarity measures the similarity between the vector representations of two documents.

A higher similarity score indicates greater textual similarity between the submitted essay and the corresponding source.

---

## 🌐 Web Crawling Pipeline

The web crawling component retrieves relevant online documents that can be used during plagiarism analysis.

```text
Essay Topic
     │
     ▼
Web Search
     │
     ▼
Relevant URLs
     │
     ▼
Web Page Retrieval
     │
     ▼
HTML Parsing
     │
     ▼
Text Extraction
     │
     ▼
Text Preprocessing
     │
     ▼
Plagiarism Detection
```

The crawling and text-processing pipeline extracts useful textual information from retrieved pages before passing it to the similarity analysis component.

---

## 🤖 AI-Assisted Essay Evaluation

The project integrates the **Google Gemini API** for automated essay evaluation.

The essay is evaluated according to the following criteria:

| Criterion                             | Weight |
| ------------------------------------- | -----: |
| Coherence & Organization              |    2.0 |
| Grammar & Syntax                      |    2.5 |
| Relevance to the Topic                |    1.5 |
| Use of Evidence & Examples            |    1.5 |
| Vocabulary & Language Variety         |    1.0 |
| Critical Thinking & Argument Strength |    1.5 |

For each criterion, the system can provide:

* Score
* Explanation
* Strengths
* Areas for improvement

The individual criterion scores are combined to produce an overall essay evaluation.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Web Application

* Streamlit

### Natural Language Processing & Machine Learning

* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity
* Text preprocessing

### Data Processing

* Pandas
* NumPy

### Web Crawling & Text Extraction

* Requests
* BeautifulSoup

### AI

* Google Gemini API

### Text Analysis

* Textstat

### Testing

* Pytest

---

## 📁 Project Structure

```text
plagiarism_ai_detection/
│
├── src/
│   ├── main.py
│   ├── ai_detect.py
│   ├── ai_detection.py
│   ├── crawl.py
│   ├── grading.py
│   ├── preprocess.py
│   ├── similarity.py
│   └── __init__.py
│
├── tests/
│
├── test_variables/
│
├── .devcontainer/
│
├── plagiarism_dataset.csv
├── enhanced_plagiarism_dataset.csv
├── chunked_plagiarism_dataset.csv
├── ai_generated_essays.csv
├── plagiarism_results.csv
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Divyank7436/plagiarism_ai_detection.git
cd plagiarism_ai_detection
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

The application requires API credentials for external services.

Create a `.env` file in the project root:

```env
SERPAPI_KEY=your_search_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder values with your own API credentials.

### ⚠️ Security

**Never commit API keys, passwords, tokens, or other sensitive credentials to GitHub.**

If you are using this repository for development, keep your `.env` file local and ensure it is included in `.gitignore`.

---

## ▶️ Running the Application

After installing the dependencies and configuring the required environment variables, run:

```bash
streamlit run src/main.py
```

The Streamlit application will start locally.

The user can then provide:

```text
Essay Topic
     +
Essay Text
```

and start the analysis.

---

## 📊 Output

The application provides multiple forms of analysis.

### Plagiarism Results

The system displays similarity information between the submitted essay and retrieved web sources.

### AI Similarity Results

The submitted essay is compared with AI-generated reference essays to calculate a similarity score.

### Essay Evaluation

The system provides criterion-wise evaluation and feedback covering:

* Organization
* Grammar
* Relevance
* Evidence
* Vocabulary
* Critical thinking

### CSV Data

Various datasets and analysis results can be stored in CSV format for further analysis and experimentation.

---


## 🔮 Future Improvements

Possible future improvements include:

* Semantic similarity using transformer-based embeddings.
* Improved paraphrase and semantic plagiarism detection.
* Sentence-level plagiarism highlighting.
* Support for PDF and DOCX document uploads.
* Improved AI-generated-content detection.
* Integration of open-source language models.
* Better web-source ranking and filtering.
* Persistent database storage.
* User authentication and analysis history.
* Detailed visual analytics.
* Automated downloadable plagiarism reports.
* Deployment as a production-ready web application.

---

## 🎓 Project Purpose

This project was developed as a final-year academic project to explore the application of:

* Natural Language Processing
* Text similarity analysis
* Web information retrieval
* Web crawling
* Machine learning techniques
* Generative AI

in the area of **academic writing analysis and plagiarism detection**.

The goal is to provide a unified platform that can analyze the originality and quality of submitted essays while providing meaningful feedback to users.

---

## 📄 License

This project was developed for academic purposes as a collaborative final-year project.

Please contact the project contributors before using the project or its datasets for commercial purposes.
