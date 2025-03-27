import os
import json
import time
from langchain_ollama.llms import OllamaLLM
import re
from langchain_core.prompts import PromptTemplate

# Constants
OLLAMA_IP = ""
models = ['gemma2']
FOLDER_PATH = "train-clpsych2025-v1"
folder_name = "default_prompt_langchain_full_train"
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
# timelines = timelines[0]  # Assuming we process the first file


# Initialize LangChain LLM Wrapper
# Initialize LangChain LLM Wrapper
def get_llm(model_name):
    return OllamaLLM(model=model_name)


# Define LangChain Prompt Templates
def create_prompt_template(task_description, response_format):
    return PromptTemplate.from_template(f"""
    {task_description}

    **Post:**
    \"{{post_text}}\"

    **Response format (strict JSON):**
    {response_format}
    """)


def query_ollama(model_name, prompt_template, post_text, max_retries=5, retry_delay=2):
    """ Queries Ollama via LangChain with retries and JSON parsing. """
    llm = get_llm(model_name)

    # Properly format the prompt with post_text
    formatted_prompt = prompt_template.format(post_text=post_text)

    for attempt in range(max_retries):
        try:
            response = llm.invoke(formatted_prompt)  # Pass the formatted string
            print("R")
            print(response)
            print()

            def extract_wellbeing_score(text):
                match = re.search(r"\b(?:well-being score of )(\b10\b|\b[1-9]\b)", text)
                return int(match.group(1)) if match else 5

            if 'well-being score of' in response and extract_wellbeing_score(response):
                return {"wellbeing_score":extract_wellbeing_score(response)}
            response_json = response[response.find("{"):1+response.find("}")]
            print('I',response_json)

            response_json = json.loads(response_json)  # Parse JSON
            print('ANS', response_json)
            return response_json  # Successfully parsed response
        except json.JSONDecodeError:
            print(f"⚠️ JSON Decode Error. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"❌ Error: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    print("❌ Max retries reached. Returning empty result.")
    return {}


# Prompt Templates
extract_evidence_template = create_prompt_template(
    task_description="""
    Given the following Reddit post, identify evidence of adaptive and maladaptive self-states.
    Extract text spans as JSON lists.
    """,
    response_format="""
    {{
      "adaptive_evidence": [<text spans that show adaptive self-states>],
      "maladaptive_evidence": [<text spans that show maladaptive self-states>]
    }}
    """
)

predict_wellbeing_template = create_prompt_template(
    task_description="""
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
    """,
    response_format="""
    {{ "wellbeing_score": <integer between 1 and 10> }}
    """
)

summarize_post_template = create_prompt_template(
    task_description="""
       Given the following Reddit post, begining by determining which self-state is dominant (adaptive/maladaptive) and describe it first then summarize the interplay between adaptive and maladaptive self-states.

    """,
    response_format="""
    {{ "summary": "<concise analysis of self-states in the post>" }}
    """
)

summarize_timeline_template = create_prompt_template(
    task_description="""
   Given the following series of Reddit posts from one user, generate a timeline-level summary.
    Focus on the interplay between adaptive and maladaptive self-states over time.
    """,
    response_format="""
    {{ "summary": "<timeline-level psychological summary>" }}
    """
)

# Main Processing Loop
for model in models:
    submission_output = {}

    # timeline = timelines
    for timeline in timelines:
        timeline_id = timeline["timeline_id"]
        submission_output[timeline_id] = {"timeline_level": {}, "post_level": {}}

        all_posts = []

        for post in timeline["posts"]:
            post_id = post["post_id"]
            post_text = post["post"]

            print('evidence')
            evidence = query_ollama(model, extract_evidence_template, post_text)
            print('adaptive_evidence')

            adaptive_evidence = evidence.get("adaptive_evidence", [])
            print('maladaptive_evidence')

            maladaptive_evidence = evidence.get("maladaptive_evidence", [])
            print('wellbeing_score')

            # Predict well-being score
            wellbeing_score = query_ollama(model, predict_wellbeing_template, post_text).get("wellbeing_score", 5)
            print('post_summary')

            # Generate post summary
            post_summary = query_ollama(model, summarize_post_template, post_text).get("summary", "")

            # Store post-level results
            submission_output[timeline_id]["post_level"][post_id] = {
                "adaptive_evidence": adaptive_evidence,
                "maladaptive_evidence": maladaptive_evidence,
                "summary": post_summary,
                "well-being score": wellbeing_score
            }

            all_posts.append(post_text)

        # Generate timeline summary
        timeline_summary = query_ollama(model, summarize_timeline_template, "\n\n".join(all_posts)).get("summary", "")
        submission_output[timeline_id]["timeline_level"]["summary"] = timeline_summary

    # Save to JSON file
    file_path = os.path.join(folder_name, f"{model}_full_train_submission.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, indent=4, ensure_ascii=False)

    print(f"✅ Submission file saved as {file_path}")
