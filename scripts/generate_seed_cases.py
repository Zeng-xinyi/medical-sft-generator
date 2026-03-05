import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/data1/zxy/models/qwen/Qwen2.5-0.5B-Instruct"

# -------------------------
# Load tokenizer & model
# -------------------------

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# -------------------------
# Competency Templates
# -------------------------

templates = {
    "Symptom Gathering": """Case:
Age:
Gender:
Chief Complaint:
History of Present Illness:
Relevant Background:

Question:
What additional symptoms should be collected?

Answer:
- Symptom 1:
  - Why it is important:
- Symptom 2:
  - Why it is important:
- Symptom 3:
  - Why it is important:
""",
    "Differential Diagnosis": """Case:
Age:
Gender:
Chief Complaint:
History of Present Illness:
Past Medical History:
Physical Examination:

Question:
Provide differential diagnoses.

Answer:
1. Diagnosis:
   - Supporting Evidence:
   - Distinguishing Features:
2. Diagnosis:
   - Supporting Evidence:
   - Distinguishing Features:
3. Diagnosis:
   - Supporting Evidence:
   - Distinguishing Features:
""",
    "Recommend Tests": """Case:
Age:
Gender:
Chief Complaint:
History:
Physical Examination:

Question:
What diagnostic tests should be recommended and why?

Answer:
1. Test:
   - Purpose:
   - How the result would influence management:
2. Test:
   - Purpose:
   - How the result would influence management:
3. Test:
   - Purpose:
   - How the result would influence management:
""",
    "Interpretation": """Case:
Age:
Gender:
Clinical Context:
Laboratory / Imaging Results:

Question:
Interpret the findings.

Answer:
1. Abnormal Findings:
2. Clinical Significance:
3. Implications for Diagnosis or Management:
""",
    "Final Diagnosis": """Case:
Age:
Gender:
Chief Complaint:
History of Present Illness:
Past Medical History:
Physical Examination:
Relevant Tests:

Question:
What is the most likely diagnosis?

Answer:
Final Diagnosis:
Step-by-Step Reasoning:
Why Alternative Diagnoses Are Less Likely:
"""
}

# -------------------------
# Load seed topics
# -------------------------

if not os.path.exists("/data1/zxy/projects/medical-sft-generator/data/seed.json"):
    print("Error: seed.json not found!")
    exit(1)

with open("/data1/zxy/projects/medical-sft-generator/data/seed.json", "r") as f:
    seeds = json.load(f)

results = []

print(f"Generating cases for {len(seeds)} seeds...")

for seed in tqdm(seeds):
    competency = seed["competency"]
    template = templates.get(competency)

    if template is None:
        continue

    prompt = (
        "Generate a medical training case.\n\n"
        "Department: " + str(seed['department']) + "\n"
        "Competency: " + str(seed['competency']) + "\n"
        "Difficulty: " + str(seed['difficulty']) + "\n"
        "Example disease: " + str(seed['disease_example']) + "\n\n"
        "Follow this format strictly:\n\n" +
        str(template) + "\n"
    )

    messages = [
        {"role": "system", "content": "You are a senior clinical medical educator."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    # -------- Generate -------- 
    

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    results.append({
        "id": seed["id"],
        "department": seed["department"],
        "competency": seed["competency"],
        "difficulty": seed["difficulty"],
        "case_text": text.strip()
    })

os.makedirs("/data1/zxy/projects/medical-sft-generator/output", exist_ok=True)

with open("/data1/zxy/projects/medical-sft-generator/output/seed_cases.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Seed cases generated successfully!")