from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai

# Load preprocessed data
with open("course_metadata.pkl", "rb") as metadata_file:
    course_metadata = pickle.load(metadata_file)

faiss_index = faiss.read_index("course_index.faiss")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
genai.configure(api_key="AIzaSyArooOBBu0OEF9uPBnzstTqfdJoRIJMItk")  # Replace with your API key

# Define generation model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "**Assistant Role Description**:\n\n"
        "You are a **smart learning assistant chatbot** embedded in a **Learning Management System (LMS)**. "
        "Your role is to provide personalized, context-aware guidance to users based on their educational goals and current progress.\n\n"
        "### **Capabilities:**\n"
        "1. **Personalized Course Guidance**: Recommend courses, suggest learning paths, and tailor responses based on user profiles.\n"
        "2. **Real-Time Support**: Resolve user doubts and engage in meaningful, conversational interactions.\n"
        "3. **Interactive Learning**: Generate quizzes and provide feedback to enhance learning.\n"
        "4. **Efficient Information Processing**: Use pre-filtered course data to keep responses relevant and concise.\n\n"
        "**Responsibilities:** Maintain context, adapt dynamically to user inputs, and make learning engaging while keeping responses concise and informative. "
        "Also, max rating is 1, which represents 100%; 0.7 and above are high ratings."
    ),
)

# FastAPI setup
app = FastAPI()

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ChatRequest(BaseModel):
    user_info: dict
    query: str
    chat_history: list  # List of {"role": "user" | "assistant", "message": "..."}

# Helper Functions
def find_relevant_courses(query, top_n=5):
    query_embedding = embedder.encode([query])[0]
    distances, indices = faiss_index.search(np.array([query_embedding]), top_n)
    return [course_metadata[idx] for idx in indices[0]]

def generate_response(user_query, relevant_courses, chat_history, user_context=None):
    course_summaries = "\n".join(
        [f"{course.get('course_title', 'Unnamed Course')} "
         f"(Rating: {course.get('rating', 'N/A')}/1, Category: {course.get('category', 'N/A')})"
         for course in relevant_courses]
    )
    
    context_info = f"User Context: {user_context}" if user_context else "No additional user context provided."
    chat_history_text = "\n".join([f"{entry['role']}: {entry['message']}" for entry in chat_history])

    prompt = (
        f"User Query: {user_query}\n\n"
        f"{context_info}\n\n"
        f"Chat History:\n{chat_history_text}\n\n"
        f"Relevant Courses:\n{course_summaries}\n\n"
        "Based on the user’s input and available courses, provide personalized guidance on the most suitable courses to take. "
        "Explain why these courses are appropriate and suggest the next steps in the learning journey."
    )

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [user_query]}]
    )
    response = chat_session.send_message(prompt)
    return response.text

# API routes
@app.post("/chatbot/")
async def chatbot_endpoint(payload: ChatRequest):
    try:
        user_query = payload.query
        user_context = payload.user_info
        chat_history = payload.chat_history

        # Retrieve relevant courses
        relevant_courses = find_relevant_courses(user_query, top_n=5)
        
        # Generate chatbot response
        response = generate_response(user_query, relevant_courses, chat_history, user_context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
