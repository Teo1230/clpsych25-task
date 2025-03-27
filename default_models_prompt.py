import os
import json
import requests

# Constants
OLLAMA_IP = ""
models = ['llama2', 'llama3.1', 'llama3.2', 'mistral', 'gemma2']
global OLLAMA_MODEL
FOLDER_PATH = "train-clpsych2025-v1"
folder_name = "default_prompt_full_train"
os.makedirs(folder_name, exist_ok=True)
import time

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
    """ Uses Gemma2 to extract adaptive and maladaptive self-state evidence from a post. """
    print("extract_evidence")
    prompt = f"""
    Given the following Reddit post, identify evidence of adaptive and maladaptive self-states.
    Extract text spans as JSON lists.

    Post:
    \"{post_text}\"

    Response format:
    {{
      "adaptive_evidence": [<adaptive text spans>],
      "maladaptive_evidence": [<maladaptive text spans>]
    }}
    """

    return query_ollama(prompt)


def predict_wellbeing(post_text):
    print("predict_wellbeing")

    """ Uses Gemma2 to predict well-being score based on extracted evidence. """
    prompt = f"""
    Given the following Reddit post, assign a well-being score from 1 (low) to 10 (high).
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
    Post:
    \"{post_text}\"

    Response format:
    {{ "wellbeing_score": <score> }}
    """

    return query_ollama(prompt)


def summarize_post(post_text):
    print("summarize_post")

    """ Uses Gemma2 to generate a post-level summary of self-states. """
    prompt = f"""
    Given the following Reddit post, summarize the interplay between adaptive and maladaptive self-states.

    Post:
    \"{post_text}\"

    Response format:
    {{ "summary": "<post-level summary>" }}
    """

    return query_ollama(prompt)


def summarize_timeline(posts):
    print("summarize_timeline")

    """ Uses Gemma2 to generate a timeline-level summary based on all posts in a timeline. """
    timeline_text = "\n\n".join(posts)

    prompt = f"""
    Given the following series of Reddit posts from one user, generate a timeline-level summary.
    Begin by determining which self-state is dominant (adaptive/maladaptive) and describe it first and focus on the interplay between adaptive and maladaptive self-states over time.

    Timeline:
    \"{timeline_text}\"

    Response format:
    {{ "summary": "<timeline-level summary>" }}
    """

    return query_ollama(prompt)


for model in models:
    OLLAMA_MODEL = model

    submission_output = {}

    for timeline in timelines:
        timeline_id = timeline["timeline_id"]
        submission_output[timeline_id] = {"timeline_level": {}, "post_level": {}}

        # Collect all posts in the timeline
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

    file_path = os.path.join(folder_name, f"{OLLAMA_MODEL}_full_timeline_submission.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, indent=4, ensure_ascii=False)

    print(f"Submission file saved as {OLLAMA_MODEL}_submission.json")
