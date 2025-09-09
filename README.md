# 🎓 Federal Board Study Bot

**Your AI-powered study companion for Federal Board textbooks in Pakistan**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Google AI](https://img.shields.io/badge/Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

## 🌟 Live Demo

**🚀 [Try the Study Bot Now](https://rag-deployed-website.streamlit.app)**

## 📚 What is this?

The Federal Board Study Bot is an intelligent RAG (Retrieval-Augmented Generation) system designed specifically for Pakistani students studying under the Federal Board curriculum. It provides instant, accurate answers to questions from official textbooks across all grades (9-12) and subjects.

### ✨ Key Features

- **📖 Textbook Integration**: Direct access to official Federal Board textbook content
- **🎯 Grade-Specific Learning**: Tailored for Grades 9, 10, 11, and 12
- **📚 Multi-Subject Support**: Mathematics, Biology, Chemistry, Physics, Computer Science
- **📍 Page References**: Shows exact page numbers where information can be found
- **❓ Dynamic Practice Questions**: AI-generated SLO-style questions for each topic
- **🤖 Student-Friendly**: Simple, encouraging explanations perfect for students
- **⚡ Fast & Reliable**: Cloud-hosted with instant responses

## 🛠️ How It Works

1. **Select Your Grade & Subject**: Choose from Grades 9-12 and any subject
2. **Ask Your Question**: Type any question about the textbook content
3. **Get Instant Answers**: Receive accurate, student-friendly explanations
4. **Find Page References**: See exactly where to find the information in your book
5. **Practice with SLO Questions**: Get AI-generated practice questions for better understanding

## 🏗️ Technical Architecture

### Tech Stack
- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: Google Gemini (LLM + Embeddings)
- **Database**: Supabase (PostgreSQL with vector support)
- **RAG Framework**: LangChain
- **OCR**: Tesseract + PyMuPDF (for scanned PDFs)
- **Deployment**: Streamlit Cloud

### System Architecture
```
Student Question → Google Gemini Embeddings → Supabase Vector Search → 
Retrieved Context → Google Gemini LLM → Student-Friendly Answer + Page References
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Supabase account

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/haniyya-h/RAG-deployed-website.git
   cd RAG-deployed-website
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 Database Schema

The system uses Supabase with the following structure:

```sql
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    grade VARCHAR NOT NULL,
    subject VARCHAR NOT NULL,
    chunk_id VARCHAR NOT NULL,
    page_content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768)
);
```

## 🔧 Environment Variables

Create a `.env` file with:

```env
GOOGLE_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_DB_URL=your_supabase_db_url
```

## 📁 Project Structure

```
RAG-deployed-website/
├── app.py                 # Main Streamlit application
├── database.py            # Supabase database operations
├── preprocess.py          # PDF processing and embedding generation
├── migrate_to_supabase.py # Migration script for local to cloud
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🎯 Supported Subjects

| Grade | Mathematics | Biology | Chemistry | Physics | Computer Science |
|-------|-------------|---------|-----------|---------|------------------|
| 9     | ✅          | ✅      | ✅        | ✅      | ✅               |
| 10    | ✅          | ✅      | ✅        | ✅      | ✅               |
| 11    | ✅          | ✅      | ✅        | ✅      | ✅               |
| 12    | ✅          | ✅      | ✅        | ✅      | ✅               |

## 🌐 Deployment

This application is deployed on **Streamlit Cloud** with the following configuration:

- **Repository**: `haniyya-h/RAG-deployed-website`
- **Branch**: `main`
- **Main file**: `app.py`
- **Database**: Supabase (cloud-hosted)
- **URL**: [https://rag-deployed-website.streamlit.app](https://rag-deployed-website.streamlit.app)

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federal Board of Pakistan** for the official curriculum
- **Google AI** for Gemini API
- **Supabase** for the database infrastructure
- **Streamlit** for the web framework
- **LangChain** for the RAG framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/haniyya-h/RAG-deployed-website/issues) page
2. Create a new issue with detailed information
3. Contact: [Your Contact Information]

## 🎓 For Students

This tool is designed to help you study more effectively. Remember:
- Always cross-reference with your physical textbook
- Use the page numbers to find detailed explanations
- Practice with the generated SLO questions
- Ask specific questions for better results

**Happy Studying! 📚✨**

---

**Made with ❤️ for Pakistani students**