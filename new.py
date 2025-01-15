
import streamlit as st
import sqlite3
import smtplib
import random
import datetime
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset

# Updated GemmaLLM class
class GemmaLLM:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # For context similarity

    def fine_tune(self, dataset_path):
        """
        Fine-tune the model using a dataset.
        
        :param dataset_path: Path to the dataset (JSON or CSV file with SQuAD-like structure).
        """
        # Load the dataset (assuming it's in SQuAD format)
        raw_datasets = load_dataset("json", data_files={"train": dataset_path})
        tokenized_datasets = raw_datasets.map(self._tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=200,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()


    def _tokenize_function(self, examples):
        """
        Tokenizes the input data for QA tasks.
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["start_positions"] = [
            self._find_start_position(context, answer["text"][0], answer["answer_start"][0])
            for context, answer in zip(examples["context"], examples["answers"])
        ]
        inputs["end_positions"] = [
            self._find_end_position(context, answer["text"][0], answer["answer_start"][0])
            for context, answer in zip(examples["context"], examples["answers"])
        ]
        return inputs

    def _find_start_position(self, context, answer_text, answer_start):
        """
        Finds the start position of the answer in the context.
        """
        return context.find(answer_text, answer_start)
    def _find_end_position(self, context, answer_text, answer_start):
        """
        Finds the end position of the answer in the context.
        """
        return self._find_start_position(context, answer_text, answer_start) + len(answer_text)

    def answer(self, question, context):
        # Preprocess context into smaller chunks if too large
        context_chunks = self.split_context(context)
        best_chunk = self.select_relevant_chunk(question, context_chunks)
        
        inputs = self.tokenizer.encode_plus(question, best_chunk, return_tensors="pt")
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )

    def split_context(self, context, max_length=512):
        # Split context into manageable chunks
        return [context[i:i+max_length] for i in range(0, len(context), max_length)]

    def select_relevant_chunk(self, question, chunks):
        # Use embeddings to find the most relevant chunk
        question_embedding = self.embedder.encode(question, convert_to_tensor=True)
        chunk_embeddings = [self.embedder.encode(chunk, convert_to_tensor=True) for chunk in chunks]
        similarities = [util.cos_sim(question_embedding, emb)[0][0].item() for emb in chunk_embeddings]
        return chunks[similarities.index(max(similarities))]

# Replace old GemmaLLM initialization
gemma_model = GemmaLLM()

# Document and Website Handling
def fetch_website_content(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver_path = "path/to/chromedriver"

    try:
        driver = webdriver.Chrome(service=Service(driver_path), options=options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        return ' '.join(p.text for p in soup.find_all('p'))
    except Exception as e:
        return f"Error fetching website content: {e}"
        
def handle_document_question(question, doc_input):
    context_chunks = gemma_model.split_context(doc_input)
    return gemma_model.answer(question, " ".join(context_chunks))

def handle_website_question(question, website_input):
    website_content = fetch_website_content(website_input)
    if "Error" in website_content:
        return website_content
    return gemma_model.answer(question, " ".join(website_content))


# Initialize Gemma LLM
gemma_model = GemmaLLM()

# Database setup
conn = sqlite3.connect("chatbot.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY, 
    password TEXT
)''')
c.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    email TEXT, 
    question TEXT, 
    answer TEXT, 
    timestamp DATETIME
)''')
conn.commit()

# Email OTP sending function
def send_otp(email):
    otp = random.randint(100000, 999999)
    st.session_state['otp'] = otp
    sender_email = "csbs22h056@gmail.com"
    sender_password = "odth hcjs wkmu zwjk"
    subject = "Your OTP Code"
    body = f"Your OTP code is {otp}."
    message = f"Subject: {subject}\n\n{body}"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, message)

# Function to get content from a website
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(p.text for p in soup.find_all('p'))
    except Exception as e:
        return f"Error fetching website content: {e}"

# Handle basic questions
def handle_basic_question(question):
    if "time" in question.lower():
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
    elif "date" in question.lower():
        return f"Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}."
    else:
        return "This is a basic question. Let me try answering: The answer may not be accurate."

# User registration
def register():
    st.subheader("Register")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
        conn.commit()
        st.success("Registration successful. Please login.")

# User login
def login():
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Send OTP"):
        c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = c.fetchone()
        if user:
            send_otp(email)
            st.session_state['email'] = email
            st.success("OTP sent to your email.")
        else:
            st.error("Invalid credentials.")
    otp = st.text_input("Enter OTP")
    if st.button("Verify OTP"):
        if otp == str(st.session_state.get('otp')):
            st.success("Login successful!")
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid OTP.")

# Chatbot interface
def chatbot():
    st.subheader("Chatbot")
    email = st.session_state.get('email')
    question = st.text_input("Ask a question")
    input_mode = st.radio("Input Source", ["Basic Question", "Document", "Website"])
    
    if input_mode == "Document":
        doc_input = st.text_area("Paste Document Text Here")
    elif input_mode == "Website":
        website_input = st.text_input("Enter Website Link Here")

    if st.button("Get Answer"):
        if input_mode == "Basic Question":
            answer = handle_basic_question(question)
        elif input_mode == "Document":
            answer = gemma_model.answer(question, doc_input)
        elif input_mode == "Website":
            website_content = fetch_website_content(website_input)
            if "Error" in website_content:
                answer = website_content
            else:
                answer = gemma_model.answer(question, website_content)
        
        st.write(f"Answer: {answer}")
        timestamp = datetime.datetime.now()
        c.execute("INSERT INTO chat_history (email, question, answer, timestamp) VALUES (?, ?, ?, ?)", 
                  (email, question, answer, timestamp))
        conn.commit()

# Chat history
def chat_history():
    st.subheader("Chat History")
    email = st.session_state.get('email')
    c.execute("SELECT question, answer, timestamp FROM chat_history WHERE email = ?", (email,))
    rows = c.fetchall()
    for row in rows:
        st.write(f"Q: {row[0]}")
        st.write(f"A: {row[1]}")
        st.write(f"Timestamp: {row[2]}")
        st.write("---")

def admin_panel():
    st.title("Admin Panel")
    st.subheader("Fine-tune Gemma LLM")

    dataset_path = st.text_input("Dataset Path", "itsupport.json")

    if st.button("Start Fine-Tuning"):
        gemma_model = GemmaLLM()
        with st.spinner("Fine-tuning in progress..."):
            gemma_model.fine_tune(dataset_path)
        st.success("Fine-tuning completed!")

# Main app
def main():
    st.title("Chatbot with 2FA and Gemma LLM")
    menu = ["Register", "Login", "Chatbot", "Chat History"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        register()
    elif choice == "Login":
        login()
    elif choice == "Chatbot":
        if st.session_state.get('logged_in'):
            chatbot()
        else:
            st.warning("Please login first.")
    elif choice == "Chat History":
        if st.session_state.get('logged_in'):
            chat_history()
        else:
            st.warning("Please login first.")

if __name__ == "__main__":
    main()