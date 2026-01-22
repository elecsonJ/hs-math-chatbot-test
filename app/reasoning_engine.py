import google.generativeai as genai
import os
from dotenv import load_dotenv
import rdflib
import json
import sys

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Try to find .env in parent directory
    parent_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Model
MODEL_NAME = "gemini-2.0-flash-exp" 
model = genai.GenerativeModel(MODEL_NAME, generation_config={"response_mime_type": "application/json"})

DEFAULT_SPARQL_PROMPT = """
    You are an expert Math Ontology Engineer.
    Your task is to convert a natural language question into a SPARQL query.
    
    ### Ontology Schema (TBox)
    {schema_info}
    
    ### Guidelines
    1. **Context**: The Ontology ONLY contains **High School Math** concepts.
    2. **Concept Mapping**:
       - High School Concept: Query directly.
       - University/Advanced Concept: **INFER** high school prerequisites (e.g., "Taylor Series" -> "Series|Differentiation") and query those.
    
    3. **Out-of-Curriculum Detection (NEW)**:
       - If the user asks about a concept that is NOT in the High School curriculum:
       - **You MUST include the exact phrase "OUT_OF_CURRICULUM" in your `explanation` field.**
       - **CRITICAL**: You MUST still generate a SPARQL query to retrieve relevant high school prerequisites. Do NOT return an empty query.
    
    4. **Output Goal**: Retrieve `Label`, `Subject`, `Chapter`. 
       - Use `FILTER(regex(?label, "Term1|Term2", "i"))`.
       - If the term might have synonyms, include them in the regex (e.g. "미분계수|순간변화율").
       - **ALWAYS use the prefix**: `PREFIX : <http://snu.ac.kr/math/>`
    
    ### Example 1 (High School Query)
    Question: "합성함수 미분이 뭐야?"
    Response:
    {{
        "query": "PREFIX : <http://snu.ac.kr/math/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?targetLabel ?targetSubject ?targetChapter WHERE {{ ?target a :Concept ; rdfs:label ?targetLabel . FILTER(regex(?targetLabel, '합성함수의 미분', 'i')) OPTIONAL {{ ?targetSection :hasConcept ?target . ?targetChapNode :hasSection ?targetSection . ?targetSubNode :hasChapter ?targetChapNode . ?targetSubNode rdfs:label ?targetSubject . ?targetChapNode rdfs:label ?targetChapter . }} }}",
        "explanation": "'합성함수의 미분'은 고교 과정에 있으므로 직접 검색합니다."
    }}
    
    ### Example 2 (University Query - Mapping)
    Question: "테일러 급수가 너무 어려워."
    Response:
    {{
        "query": "PREFIX : <http://snu.ac.kr/math/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?targetLabel ?targetSubject ?targetChapter WHERE {{ ?target a :Concept ; rdfs:label ?targetLabel . FILTER(regex(?targetLabel, '급수|합성함수의 미분|이계도함수', 'i')) OPTIONAL {{ ?targetSection :hasConcept ?target . ?targetChapNode :hasSection ?targetSection . ?targetSubNode :hasChapter ?targetChapNode . ?targetSubNode rdfs:label ?targetSubject . ?targetChapNode rdfs:label ?targetChapter . }} }}",
        "explanation": "'테일러 급수'는 온톨로지에 없으므로, 이를 이해하기 위해 필요한 고교 과정인 '급수', '합성함수의 미분', '이계도함수'를 검색합니다."
    }}
    
    ### User Question
    {question}
    
    ### Response
    """

DEFAULT_ANSWER_PROMPT = """
    You are a Math Mentor Chatbot.
    
    ### User Question
    {question}
    
    ### Retrieved Knowledge (SPARQL Results)
    {data_summary}
    (Logic: {sparql_explanation})
    
    ### Instructions
    1. **Analyze**: Carefully evaluate the retrieved data (Concepts, Prerequisites, etc.) and the provided Logic string.
    
    2. **Scope & Ambiguity Check (CRITICAL)**:
       - **Case A: Out of Curriculum** (Logic contains "OUT_OF_CURRICULUM"):
         - Start answer with: "교육과정 외의 내용입니다."
         - Explain that the concept is advanced and link it to the retrieved High School prerequisites.
         - **MUST provide the 'evidence' list** containing those prerequisites.
       
       - **Case B: Concept Ambiguity** (Same Name, Different Depth):
         - If the user asks about a concept (e.g., "Continuous Probability Distribution", "Matrix") that exists in High School but implies a University-level depth (e.g., "Is this all?", "General definition"):
         - **Do NOT** simply say "Study the high school version."
         - Explicitly clarify: "고등학교 과정에서는 ~만 다루지만, 대학 과정에서는 ~까지 확장됩니다." (Distinguish the scope).
         - Then, guide them to the High School concepts available in the ontology.

    3. **Answer Style & Language**: 
       - **Language**: Write the entire 'answer' in **Korean**.
       - **Tone**: Maintain an encouraging, empathetic, and helpful mentor persona.
       - **Guidance**: Use the `Retrieved Knowledge` to suggest which high school foundations the student should review.
    
    4. **Evidence Construction**:
       - Create an 'evidence' list based strictly on the 'Retrieved Knowledge'.
       - **IMPORTANT**: Even if the concept is "Out of Curriculum" or "Ambiguous", you MUST list the retrieved related concepts in `evidence` so they can be visualized.
       - Map the data to: subject, chapter, concept, and desc (short reason for relevance).
       - If hierarchy info (subject/chapter) is missing, infer it from the context or use "Unknown".
       
    ### Output Format (JSON)
    Strictly adhere to this Typescript Interface:
    interface Response {{
        answer: string; // Must start with "교육과정 외의 내용입니다." if applicable.
        evidence: {{
            subject: string;
            chapter: string;
            concept: string;
            desc?: string; // e.g., "Prerequisite for this advanced topic"
        }}[];
    }}
    """

def generate_sparql(question, schema_info, prompt_template=DEFAULT_SPARQL_PROMPT):
    try:
        # Check if placeholders exist, if so use format, if not just append (fallback)
        prompt = prompt_template.replace("{schema_info}", str(schema_info)).replace("{question}", question)
    except Exception as e:
        return {"query": "", "explanation": f"Prompt Formatting Error: {e}"}
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"[ERROR] SPARQL Generation Failed: {e}")
        return {"query": "", "explanation": f"Error: {e}"}

def execute_sparql(query, graph):
    """
    Executes the SPARQL query on the given graph.
    """
    try:
        results = graph.query(query)
        data = []
        for row in results:
            item = {}
            for var in results.vars:
                val = row[var]
                item[str(var)] = str(val) if val is not None else None
            data.append(item)
        return data
    except Exception as e:
        print(f"[ERROR] SPARQL Execution Failed: {e}")
        return []

def generate_answer(question, raw_data, sparql_explanation, prompt_template=DEFAULT_ANSWER_PROMPT):
    """
    Generates a structured JSON answer with 'answer' and 'evidence'.
    """
    
    data_summary = json.dumps(raw_data, ensure_ascii=False) if raw_data else "No data found."
    
    prompt = prompt_template.replace("{question}", question)\
                            .replace("{data_summary}", data_summary)\
                            .replace("{sparql_explanation}", str(sparql_explanation))
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"[ERROR] Answer Generation Failed: {e}")
        return {
            "answer": f"답변 생성 중 오류가 발생했습니다. ({e})",
            "evidence": []
        }
