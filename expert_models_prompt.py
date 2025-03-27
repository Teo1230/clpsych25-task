import os
import json
import requests
import time

# Constants
OLLAMA_IP = ""
models = ['llama2', 'llama3.1', 'llama3.2', 'mistral', 'gemma2']
global OLLAMA_MODEL
FOLDER_PATH = "test-clpsych2025"
folder_name = "expert_prompt_test"
os.makedirs(folder_name, exist_ok=True)

# Function to read JSON files
def read_json_files(folder):
    structured_data = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    structured_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}")
    return structured_data

# Load Data
timelines = read_json_files(FOLDER_PATH)


def query_ollama(prompt, max_retries=5, retry_delay=2):
    """ Sends a request to Ollama API and ensures complete response with error handling. """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_IP}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False},
                headers={"Content-Type": "application/json"},
                timeout=30  # Avoid indefinite hanging
            )
            response.raise_for_status()  # Raise an error for bad responses (e.g., 500, 404)

            response_json = response.json()

            raw_response = response_json.get("response", "")

            try:
                parsed_response = json.loads(raw_response)
                return parsed_response  # Successfully parsed response
            except json.JSONDecodeError:
                print(f"JSON Decode Error. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue  # Retry

        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    print("Max retries reached. Returning empty result.")
    return {}

def extract_evidence(post_text):
    print('extract_evidence')
    """ Identifies adaptive and maladaptive self-state evidence in a post. """
    prompt = f"""
    You are an expert in **psychological self-states and mental health analysis**. Your task is to analyze the Reddit post below and extract textual evidence that indicates **adaptive and maladaptive self-states**. 

    - **Adaptive self-states**: Indicate resilience, coping, self-awareness, or positive cognitive and behavioral patterns.
    - **Maladaptive self-states**: Indicate distress, negative cognitive distortions, emotional dysregulation, or harmful behaviors.

    **Post:**
    \"{post_text}\"

    **Response format (strict JSON):**
    {{
      "adaptive_evidence": [<text spans that show adaptive self-states>],
      "maladaptive_evidence": [<text spans that show maladaptive self-states>]
    }}
    """
    return query_ollama(prompt)


def predict_wellbeing(post_text):
    print('predict_wellbeing')

    """ Predicts well-being score based on psychological markers in the post. """
    prompt = f"""
    You are a clinical expert in **mental health assessment**. Your task is to assign a **well-being score (1-10)** to the Reddit post below based on its emotional, cognitive, and behavioral indicators.

    - **1**: The person is in persistent danger of severely hurting self or others or persistent inability to maintain minimal personal hygiene or has attempted a serious suicidal act with a clear expectation of death.
    - **2**: In danger of hurting self or others (eg., suicide attempts; frequently violent; manic excitement) or may fail to maintain minimal personal hygiene or significant impairment in communication (e.g., incoherent or mute).
    - **3**: A person experiences delusions or hallucinations or serious impairment in communication or judgment or is unable to function in almost all areas (eg., no job, home, or friends).
    - **4**: Some impairment in reality testing or communication, or major impairment in multiple areas (withdrawal from social ties, inability to work, neglecting family, severe mood/thought impairment).
    - **5**: Serious symptoms (e.g., suicidal thoughts, severe compulsions) or serious impairment in social, occupational, or school functioning (eg., no friends, inability to keep a job).
    - **6**: Moderate symptoms (eg., panic attacks) or moderate difficulty in social, occupational or school functioning.
    - **7**: Mild symptoms (eg., depressed mood and mild insomnia) or some difficulty in social, occupational, or school functioning, but generally functioning well, has some meaningful interpersonal relationships.
    - **8**: If symptoms are present, they are temporary and expected reactions to psychosocial stressors (eg., difficulty concentrating after family argument). Slight impairment in social, occupational or school functioning.
    - **9**: Absent or minimal symptoms (eg., mild anxiety before an exam), good functioning in all areas, interested and involved in a wide range of activities.
    - **10**: No symptoms and superior functioning in a wide range of activities.
    **Post:**
    \"{post_text}\"

    **Response format (strict JSON):**
    {{ "wellbeing_score": <integer between 1 and 10> }}
    """
    return query_ollama(prompt)


def summarize_post(post_text):
    print('summarize_post')
    """ Summarizes the self-state interactions within a post. """
    prompt = f"""
    You are a **psychological expert analyzing self-states** in text. Your task is to **summarize beging by determining which self-state is dominant (adaptive/maladaptive) and describe it first then how adaptive and maladaptive self-states interact within this post**. 

    - Identify **key emotional, cognitive, and behavioral patterns**.
    - Highlight **contrasts between adaptive and maladaptive self-states**.
    - Provide an **objective, clinical-style summary**.

    **Post:**
    \"{post_text}\"

    **Response format (strict JSON):**
    {{ "summary": "<concise analysis of self-states in the post>" }}
    """
    return query_ollama(prompt)


def summarize_timeline(posts):
    print('summarize_timeline')

    """ Generates a timeline-level summary for a user's post history. """
    timeline_text = "\n\n".join(posts)

    prompt = f"""
    You are a **clinical psychologist analyzing mental health trends over time**. Given the following series of Reddit posts from a single user, summarize their **self-state trajectory**.

    - Identify **patterns of emotional and cognitive change**.
    - Note **shifts between adaptive and maladaptive self-states**.
    - Highlight **any signs of improvement, deterioration, or instability**.

    **Timeline:**
    \"{timeline_text}\"

    **Response format (strict JSON):**
    {{ "summary": "<timeline-level psychological summary>" }}
    """
    return query_ollama(prompt)


for model in models:
    OLLAMA_MODEL = model

    submission_output = {}

    for timeline in timelines:
        timeline_id = timeline["timeline_id"]
        submission_output[timeline_id] = {"timeline_level": {}, "post_level": {}}

        all_posts = []

        for post in timeline["posts"]:
            post_id = post["post_id"]
            post_text = post["post"]

            # Extract evidence
            evidence = extract_evidence(post_text)
            adaptive_evidence = evidence.get("adaptive_evidence", [])
            maladaptive_evidence = evidence.get("maladaptive_evidence", [])

            # Predict well-being score
            wellbeing_score = predict_wellbeing(post_text).get("wellbeing_score", 5)  # Default 5 if missing

            # Generate post summary
            post_summary = summarize_post(post_text).get("summary", "")

            # Store post-level results
            submission_output[timeline_id]["post_level"][post_id] = {
                "adaptive_evidence": adaptive_evidence,
                "maladaptive_evidence": maladaptive_evidence,
                "summary": post_summary,
                "well-being score": wellbeing_score
            }

            all_posts.append(post_text)

        # Generate timeline summary
        timeline_summary = summarize_timeline(all_posts).get("summary", "")
        submission_output[timeline_id]["timeline_level"]["summary"] = timeline_summary

    # Save to JSON file
    file_path = os.path.join(folder_name, f"{OLLAMA_MODEL}_begin_submission.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, indent=4, ensure_ascii=False)

    print(f"Submission file saved as {file_path}")
