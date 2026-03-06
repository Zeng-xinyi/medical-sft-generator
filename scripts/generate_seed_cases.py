import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 1. 路径配置
# =========================

MODEL_NAME = "/data1/zxy/models/qwen/Qwen2.5-3B-Instruct"
SEED_FILE = "/data1/zxy/projects/medical-sft-generator/data/seed.json"
OUTPUT_FILE = "/data1/zxy/projects/medical-sft-generator/output/sft_train_data.json"

# =========================
# 2. 加载模型
# =========================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# =========================
# 3. 模板定义
# =========================

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

# =========================
# 4. 文本解析函数
# =========================

def parse_generated_text(text):

    try:

        # 去掉markdown
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+', '', text)

        q_match = re.search(r'Question\s*:', text, re.IGNORECASE)
        a_match = re.search(r'Answer\s*:', text, re.IGNORECASE)

        if not q_match or not a_match:
            return None, None

        case_part = text[:q_match.start()].strip()
        question_part = text[q_match.end():a_match.start()].strip()
        answer_part = text[a_match.end():].strip()

        instruction = f"{case_part}\n\nQuestion:\n{question_part}"

        return instruction, answer_part

    except:
        return None, None

# =========================
# 5. 读取 seed
# =========================

if not os.path.exists(SEED_FILE):
    print("Seed file not found")
    exit()

with open(SEED_FILE) as f:
    seeds = json.load(f)

print(f"Generating cases for {len(seeds)} seeds...")

sft_data = []

# =========================
# 6. 生成数据
# =========================

debug_count = 0

for seed in tqdm(seeds):

    competency = seed["competency"]
    template = templates.get(competency)

    if template is None:
        continue

    prompt = f"""
Generate ONE medical clinical training example.

Requirements:
- Realistic clinical scenario
- Match department and disease
- No markdown
- No extra text
- Follow structure exactly

Department: {seed['department']}
Competency: {seed['competency']}
Difficulty: {seed['difficulty']}
Disease: {seed['disease_example']}

Output format:

{template}

Important:
The output MUST include Case, Question, and Answer sections.
"""

    messages = [
        {"role": "system", "content": "You are a senior medical educator."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=700,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.05,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Debug输出
    if debug_count < 3:
        print("\n------RAW OUTPUT------")
        print(text)
        print("----------------------")
        debug_count += 1

    # =========================
    # 如果没有Answer 自动补全
    # =========================

    if "Answer:" not in text:

        fix_prompt = f"""
Complete the medical case by adding ONLY the missing Answer section.

{text}

Answer:
"""

        messages = [
            {"role": "system", "content": "You are a clinical expert."},
            {"role": "user", "content": fix_prompt}
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3
        )

        fix_ids = outputs[0][inputs["input_ids"].shape[1]:]
        fix_text = tokenizer.decode(fix_ids, skip_special_tokens=True)

        text = text + "\nAnswer:\n" + fix_text

    # =========================
    # 解析
    # =========================

    instruction, response = parse_generated_text(text)

    if instruction and response:

        sft_data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "meta": {
                "department": seed["department"],
                "competency": seed["competency"],
                "difficulty": seed["difficulty"]
            }
        })

    else:

        print(f"Warning: Failed to parse seed {seed['id']}")

# =========================
# 7. 保存数据
# =========================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, indent=2, ensure_ascii=False)

print("\nGeneration finished")
print(f"Total samples: {len(sft_data)}")
print(f"Saved to: {OUTPUT_FILE}")