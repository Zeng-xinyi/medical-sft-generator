import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

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
"""
}

# -------------------------
# Load seed topics
# -------------------------

with open("data/seed.json","r") as f:
    seeds = json.load(f)

results = []

for seed in tqdm(seeds):

    competency = seed["competency"]

    template = templates.get(competency)

    if template is None:
        continue

    prompt = f"""
You are a senior clinical medical educator.

Generate a medical training case.

Department: {seed['department']}
Competency: {seed['competency']}
Difficulty: {seed['difficulty']}
Example disease: {seed['disease_example']}

Follow this format strictly:

{template}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({
        "id": seed["id"],
        "department": seed["department"],
        "competency": seed["competency"],
        "difficulty": seed["difficulty"],
        "case_text": text
    })

with open("output/seed_cases.json","w") as f:
    json.dump(results,f,indent=2)

print("Seed cases generated!")