# 🧬 BioMed-QA: Biomedical Question-Answering System (NLP-Project)
An intelligent Question-Answering system for biomedical research. This interactive app uses state-of-the-art NLP models like BioBERT to search thousands of PubMed abstracts and extract direct answers to complex scientific questions, moving beyond simple keyword search.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) An advanced, domain-specific Question-Answering (QA) system designed to navigate the vast world of biomedical research literature. This interactive application, built with Streamlit, leverages state-of-the-art NLP models to provide direct answers to complex questions, moving far beyond traditional keyword search.

The system features a full comparative framework to evaluate multiple Information Retrieval (IR) and Named Entity Recognition (NER) techniques, complete with quantitative performance metrics (P@k, R@k, nDCG@k, F1-Score).



---
## 🚀 Key Features

* **Multi-Engine Information Retrieval:** Compare four different search methods:
    * **TF-IDF:** Classic keyword-based search.
    * **Word2Vec:** Baseline semantic search.
    * **Dense (BioBERT):** State-of-the-art semantic search using a model fine-tuned on PubMed.
    * **Hybrid Search:** A powerful blend of Dense and TF-IDF search, getting the best of both worlds.
* **Dual-System Named Entity Recognition (NER):**
    * **BioNER Model:** A powerful Hugging Face transformer model for high-accuracy entity extraction (`DISEASE`, `CHEMICAL`, `GENE`).
    * **Regex NER:** A robust, rule-based fallback system.
* **Quantitative Evaluation:** In-app tabs to evaluate and visualize the performance of both IR and NER components with standard academic metrics.
* **Interactive NLP Demos:** Dedicated sections to demonstrate fundamental NLP concepts like Part-of-Speech (POS) Tagging and Word Sense Disambiguation (WSD).
* **Fully Interactive UI:** Built with Streamlit for a clean, user-friendly experience.

---
## 🏛️ System Architecture

The project follows a two-stage **Retrieve-then-Extract** pipeline to efficiently find answers.



1.  **Information Retrieval Engine:** A user's query is first sent to the IR engine, which uses one of the four search methods to quickly find the top-K most relevant documents from a database of thousands of abstracts.
2.  **Answer Extraction Engine:** These few relevant documents are then passed to the NER engine, which performs a deep analysis of the text to identify and extract the specific entities that answer the user's question.

---
## 🛠️ Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/biomed-qa.git](https://github.com/your-username/biomed-qa.git)
cd biomed-qa
