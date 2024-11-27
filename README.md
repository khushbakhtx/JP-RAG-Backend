# **JP-RAG System**

**JP-RAG** is a Retrieval-Augmented Generation (RAG) project designed to interact with and provide insights from the books *12 Rules for Life* and *Beyond Order: 12 More Rules* by Jordan Peterson. This system uses modern AI techniques to retrieve relevant content from the books and generate meaningful, context-aware responses to user queries.

---

## **Features**
- **Conversational AI**: A chat-like interface for seamless user interaction.
- **Efficient Retrieval**: Uses semantic search to retrieve relevant content from the books based on user queries.
- **Contextual Generation**: Leverages a language model to generate insightful answers based on retrieved content.
- **Interactive UI**: A React-based frontend for intuitive and engaging user experiences.
- **Keyboard Support**: Supports both "Send" button and **Enter** key for submitting questions.

---

## **How It Works**
1. **Data Preparation**:
   - Books were chunked into smaller, meaningful sections (each rule was treated as a content chunk).
   - These chunks were embedded using **HuggingFace embeddings** for semantic search.
   
2. **RAG Workflow**:
   - **Retriever**: Searches for the most relevant chunks from the vector store based on user input.
   - **Generator**: Generates a detailed and contextual response using a language model (LLM).
   
3. **Frontend**:
   - Built with **React** to provide a dynamic chat interface.
   - Displays conversation history and allows users to toggle between various views (Chat, Help, Activity, Settings).

4. **Backend**:
   - Developed using **FastAPI**, providing endpoints for handling queries.
   - Manages the RAG pipeline and communicates with the React frontend.

---

## **Technologies Used**
### **Backend**
- **FastAPI**: Lightweight Python framework for building APIs.
- **Chroma/FAISS**: Vector store for efficient semantic search.
- **HuggingFace Transformers**: For generating embeddings and leveraging pre-trained models.
- **Sentence-Transformers**: Used for embedding text chunks.

### **Frontend**
- **React**: JavaScript library for building the user interface.
- **Axios**: For making API calls to the backend.

### **Other Tools**
- **HuggingFace API**: Used for text generation in the RAG pipeline.
- **Python**: Primary language for backend development.
- **Markdown**: For documentation.

---

## **Setup Instructions**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/jp-rag.git
cd jp-rag
```

### **2. Install Backend Dependencies**
```bash
cd backend
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### **3. Start the Backend Server**
```bash
uvicorn app.main:app --reload
```

### **4. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

### **5. Start the Frontend Server**
```bash
npm start
```

---

## **Key Functionalities**
### **Retriever**
- The retriever searches for the most relevant content from the vector store using semantic embeddings.
- Embeddings are generated using the **"all-MiniLM-L6-v2"** model from HuggingFace.

### **Generator**
- The generator utilizes an LLM **"meta-llama/Llama-3.2-3B-Instruct"** via HuggingFace API to generate contextually relevant responses.

### **Frontend**
- Provides a chat interface for interacting with the RAG system.
- Supports toggling between different views:
  - **Chat**: Main interface for querying the RAG system.
  - **Help**: Offers guidance for using the system.
  - **Activity**: Displays recent interactions.
  - **Settings**: Customizable options for the user.

---

## **Folder Structure**
```
jp-rag/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── models.py         # Pydantic models for query/request
│   │   ├── rag_pipeline.py   # RAG pipeline logic
│   │   ├── vectorstore.py    # Vector store setup and retrieval
│   │   └── utils.py          # Helper functions
│   ├── data/                 # Prepared book data in JSON format
│   ├── embeddings/           # Embeddings and vector store files
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar/      # Sidebar component
│   │   │   ├── Main/         # Main chat interface
│   │   │   └── App.js        # Root React component
│   ├── public/               # Static files
│   └── package.json          # Frontend dependencies
├── README.md                 # Documentation
└── LICENSE                   # License file
```

---

## **Challenges Faced**
1. **Slow Response Time**: Initially, responses were delayed due to the gradual typing effect. Resolved by optimizing the display logic to show the full response immediately while retaining typing effects.
2. **Chunking the Books**: Balancing chunk size for efficient retrieval without losing context was a challenge.
3. **Frontend-Backend Communication**: Ensured seamless integration between React and FastAPI using Axios.

---

## Golden Set
Here is a list of **50 meaningful questions** derived from *Beyond the Order: 12 More Rules for Life* by Jordan Peterson. These questions cover key themes, ideas, and discussions from the book. You can use them as a **golden set** to test your Retrieval-Augmented Generation (RAG) system.

---

### **Rule 1: Do not carelessly denigrate social institutions or creative achievement**
1. Why does Jordan Peterson emphasize the importance of preserving social institutions?
2. How does Peterson explain the balance between order and chaos in society?
3. What is the significance of respecting creative achievement in the modern world?
4. How does Peterson relate social institutions to personal stability?
5. Why does Peterson warn against carelessly criticizing systems we do not fully understand?

---

### **Rule 2: Imagine who you could be and then aim single-mindedly at that**
6. What does it mean to "imagine who you could be"?
7. How can focusing on a clear goal improve one's life trajectory?
8. What role does self-improvement play in this rule?
9. How does Peterson explain the importance of single-minded focus on personal growth?
10. What advice does Peterson give for overcoming obstacles while pursuing an ideal?

---

### **Rule 3: Do not hide unwanted things in the fog**
11. What does "the fog" represent in this rule?
12. Why is it important to confront and address hidden truths in our lives?
13. How does avoiding clarity lead to unnecessary suffering?
14. What strategies does Peterson suggest for facing the "fog" in one's life?
15. How can this rule be applied in personal relationships?

---

### **Rule 4: Notice that opportunity lurks where responsibility has been abdicated**
16. How can identifying abdicated responsibilities lead to personal growth?
17. Why does Peterson believe that opportunities often lie in places others avoid?
18. How does taking responsibility for neglected areas create meaning in life?
19. What role does responsibility play in developing character?
20. How can this rule be applied in professional settings?

---

### **Rule 5: Do not do what you hate**
21. How does Peterson define actions that we "hate" doing?
22. Why is it important to align one's actions with personal values?
23. What are the consequences of consistently engaging in activities we despise?
24. How can someone recognize when they are doing something they hate?
25. How does this rule relate to integrity and authenticity?

---

### **Rule 6: Abandon ideology**
26. How does Peterson define "ideology" in this rule?
27. Why does Peterson caution against ideological thinking?
28. What are the dangers of simplifying complex realities through ideologies?
29. How can someone recognize when they are adopting an ideological perspective?
30. What is the alternative to ideology according to Peterson?

---

### **Rule 7: Work as hard as you possibly can on at least one thing and see what happens**
31. How does hard work on a meaningful task transform one's life?
32. Why does Peterson emphasize working on "at least one thing"?
33. How does mastery in a specific area impact other aspects of life?
34. What advice does Peterson give for maintaining discipline in work?
35. How does this rule relate to personal fulfillment?

---

### **Rule 8: Try to make one room in your home as beautiful as possible**
36. Why does Peterson emphasize creating beauty in one's home environment?
37. How does this rule help bring order to chaos in everyday life?
38. What is the relationship between physical surroundings and mental well-being?
39. How can this rule serve as a metaphor for improving one's life?
40. What practical steps does Peterson suggest for beautifying a space?

---

### **Rule 9: If old memories still make you cry, write them down carefully and completely**
41. How does writing about painful memories help resolve lingering emotions?
42. What therapeutic benefits does Peterson attribute to journaling about the past?
43. Why is it important to carefully articulate old memories instead of ignoring them?
44. How can confronting past traumas lead to personal growth?
45. What role does storytelling play in processing emotional pain?

---

### **Rule 10: Plan and work diligently to maintain the romance in your relationship**
46. Why is maintaining romance important for long-term relationships?
47. What practical advice does Peterson give for keeping romance alive?
48. How can planning and effort strengthen emotional bonds between partners?
49. What are the consequences of neglecting romance in a relationship?
50. How does this rule apply to modern challenges in relationships?

---

### **Using These Questions**
- **Golden Set Testing**: Use these questions to evaluate the accuracy and contextual understanding of your RAG system.
- **Diversity of Themes**: The questions cover a variety of topics like personal growth, relationships, responsibility, and societal insights, which will comprehensively test your system's retrieval and generation capabilities.
- **Performance Measurement**: Assess how well your RAG system retrieves relevant content and generates meaningful, accurate responses.
