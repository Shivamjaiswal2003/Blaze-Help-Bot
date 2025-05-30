# 🤖 Blaze AI Telegram Support Bot

An intelligent Telegram bot built to answer product-related queries for Blaze users 24/7 using natural language understanding and vector search.

---

## ✨ Features

- ✅ Automatically answers common Blaze support questions  
- 📄 Uses Blaze Gitbook & Help Docs as knowledge base  
- ⚡ Fast response using Sentence Transformers + FAISS  
- 🧠 Basic memory for recent interactions (contextual recall)  
- 🛠 Easily extendable and customizable  

---

## 📁 Project Structure

my-bot/
├── bot.py # Main script to run the bot
├── blaze_docs.txt # Knowledge base extracted from Blaze documentation
├── .env # Stores Telegram bot token (DO NOT COMMIT)
├── .gitignore # Ignores env and token files
└── venv/ # Python virtual environment (excluded)


---

## ⚙️ Setup Instructions

 📥 Clone this repository

```bash
git clone https://github.com/yourusername/blaze-telegram-bot.git
cd blaze-telegram-bot

🧪 Create and activate a virtual environment

bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


🧪 Create and activate a virtual environment

bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
📦 Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
🔐 Add your Telegram Bot token in a .env file:

ini
Copy
Edit
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
▶️ Run the bot

bash
Copy
Edit
python bot.py
💡 How It Works
Embeds Blaze documentation using SentenceTransformers

Converts incoming queries to vectors

Finds the most relevant answer using FAISS vector search

Responds to the user in real-time in Telegram chats

💬 Sample Interaction
User in Telegram group:

"How can I start a Twitter DM campaign?"

Bot responds:

"To create a Twitter DM campaign, go to Campaigns > Twitter DMs. You can segment users by wallet, engagement, or activity."

🔍 Evaluation & Future Enhancements
🛑 Risks:

Hallucinated or incomplete answers

Lack of multi-turn memory

📏 Success Metrics:

Response accuracy

User feedback or satisfaction

Reduced manual support queries

🚀 Future Improvements:

Escalation system to notify humans

Confidence-based fallback replies

RAG architecture or LLM integration

🧾 License
MIT License. Use freely and adapt as needed.

🔗 References
🔗 Blaze Gitbook: https://docs.withblaze.app/blaze

🆘 Blaze Help Center: https://intercom.help/blaze-3d9c6d1123fd/en/

🧠 Sentence Transformers: https://www.sbert.net

📚 FAISS: https://github.com/facebookresearch/faiss