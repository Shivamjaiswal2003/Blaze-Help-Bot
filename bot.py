import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Blaze documentation as knowledge base
def load_blaze_docs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into Q&A pairs
    qa_pairs = []
    current_q = None
    current_a = []
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Q: '):
            if current_q is not None:
                qa_pairs.append((current_q, ' '.join(current_a)))
            current_q = line[3:].strip()
            current_a = []
        elif line.startswith('A: '):
            current_a.append(line[3:].strip())
    
    if current_q is not None:
        qa_pairs.append((current_q, ' '.join(current_a)))
    
    return qa_pairs

# Load documents
qa_pairs = load_blaze_docs('blaze_docs.txt')
documents = [qa[1] for qa in qa_pairs]  # Using answers as documents
questions = [qa[0] for qa in qa_pairs]  # For reference

# Create FAISS index
doc_embeddings = model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm your Blaze assistant. Ask me anything about Blaze and I'll try to help!")

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    
    # Search for most relevant answer
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding), k=1)
    
    best_match_idx = I[0][0]
    best_match_answer = documents[best_match_idx]
    best_match_question = questions[best_match_idx]
    
    response = f"Q: {best_match_question}\n\nA: {best_match_answer}"
    await update.message.reply_text(response)

def main():
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    print("Bot is running... Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()