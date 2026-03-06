import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# 配置路径
# -------------------------
MODEL_NAME = "/data1/zxy/models/qwen/Qwen2.5-0.5B-Instruct"
SEED_FILE = "/data1/zxy/projects/medical-sft-generator/data/seed.json"
OUTPUT_FILE = "/data1/zxy/projects/medical-sft-generator/output/sft_train_data.json"

# -------------------------
# 1. 加载模型
# -------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# -------------------------
# 2. 构造英文 One-Shot Example
# -------------------------
# 这是根据你提供的“食管癌放疗后突发脑梗死”病例改编的标准英文范例
# 我们把它做得结构非常清晰，强迫模型模仿
ONE_SHOT_EXAMPLE = """
Example Input:
Department: Neurology
Competency: Final Diagnosis
Disease: Cerebral Infarction

Example Output:
Case:
A 73-year-old male farmer presents with sudden left-sided limb weakness for 1 day.
History: Diagnosed with esophageal cancer in 2018 (post-radiotherapy), Hypertension for 5 years (BP up to 180/110 mmHg).
Physical Exam: BP 177/107 mmHg. Conscious. Left limb muscle strength Grade 4, sensation decreased. 
Imaging: CT shows right paraventricular cerebral infarction.

Question:
What is the most likely diagnosis?

Answer:
Final Diagnosis: Cerebral Infarction; Hypertension Grade 3 (Very High Risk); Post-esophageal cancer radiotherapy.
Reasoning: The patient has risk factors (HTN, cancer history), sudden onset of focal neurological deficits, and CT confirmation.
"""

# -------------------------
# 3. 定义任务模板
# -------------------------
templates = {
    "Symptom Gathering": """Case:
[Patient details and chief complaint]

Question:
What additional symptoms should be collected?

Answer:
[List symptoms and reasons]""",

    "Differential Diagnosis": """Case:
[History and physical exam]

Question:
Provide differential diagnoses.

Answer:
[List diagnoses with evidence]""",

    "Recommend Tests": """Case:
[History and physical exam]

Question:
What diagnostic tests should be recommended?

Answer:
[List tests and purposes]""",

    "Interpretation": """Case:
[Clinical context and lab results]

Question:
Interpret the findings.

Answer:
[Analysis of findings]""",

    "Final Diagnosis": """Case:
[Full case details]

Question:
What is the most likely diagnosis?

Answer:
[Final diagnosis and reasoning]"""
}

# -------------------------
# 4. 解析函数 (兼容 Markdown)
# -------------------------
def parse_generated_text(text):
    """
    极度宽容的解析逻辑
    """
    # 1. 找 Question (允许 Question:, **Question**, ### Question 等)
    parts = re.split(r'(?i)(?:^|\n)[\#\*]*\s*Question[:\s]*', text, maxsplit=1)
    
    if len(parts) < 2:
        return None, None, "No 'Question' found"
        
    case_part = parts[0].strip()
    remaining = parts[1]
    
    # 清理 Case 部分可能存在的 "Case:" 头
    case_part = re.sub(r'(?i)^[\#\*\s]*Case[:\s]*', '', case_part).strip()
    
    # 2. 找 Answer
    q_and_a = re.split(r'(?i)(?:^|\n)[\#\*]*\s*Answer[:\s]*', remaining, maxsplit=1)
    
    if len(q_and_a) < 2:
        return None, None, "No 'Answer' found"
        
    question_part = q_and_a[0].strip()
    answer_part = q_and_a[1].strip()
    
    # 3. 组装
    full_instruction = f"Case:\n{case_part}\n\nQuestion:\n{question_part}"
    
    return full_instruction, answer_part, "Success"

# -------------------------
# 5. 生成流程
# -------------------------

if not os.path.exists(SEED_FILE):
    print(f"Error: {SEED_FILE} not found!")
    exit(1)

with open(SEED_FILE, "r") as f:
    seeds = json.load(f)

sft_data = []

print(f"Generating cases for {len(seeds)} seeds...")

for seed in tqdm(seeds):
    competency = seed["competency"]
    template = templates.get(competency)

    if template is None:
        continue

    # 构造 Prompt：One-Shot + Current Task
    # 重点：要求 Concise (简洁)，防止模型废话太多被截断
    prompt = (
        "You are a medical exam generator. Imitate the example below strictly.\n"
        "Keep the 'Case' description concise and professional.\n\n"
        f"{ONE_SHOT_EXAMPLE}\n\n"
        "Now generate a case for the following:\n"
        f"Department: {seed['department']}\n"
        f"Competency: {seed['competency']}\n"
        f"Difficulty: {seed['difficulty']}\n"
        f"Disease: {seed['disease_example']}\n\n"
        "Output:\n"
        f"{template}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Follow the format strictly. Do not use Markdown headers."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    # -------- Generate -------- 
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,   # 足够长，防止截断
            temperature=0.3,       # 重要！降低温度，让模型更听话，不做随机发散
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1
        )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # -------- Parse --------
        instruction, response, status = parse_generated_text(text)

        if status == "Success":
            sft_data.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "meta": seed
            })
        else:
            tqdm.write(f"\n[WARNING] Seed {seed['id']} failed: {status}")
            # 即使失败也保存，后续可清洗
            sft_data.append({
                "instruction": "FORMAT_ERROR",
                "input": "",
                "output": text,
                "meta": seed,
                "error": status
            })

    except Exception as e:
        print(f"Error processing seed {seed['id']}: {e}")

# -------------------------
# 6. 保存结果
# -------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
    json.dump(sft_data, f, indent=2, ensure_ascii=False)

# 统计成功率
success_count = len([x for x in sft_data if x.get("instruction") != "FORMAT_ERROR"])
print(f"\nProcessing complete!")
print(f"Valid formatted data: {success_count} / {len(sft_data)}")
print(f"Saved to: {OUTPUT_FILE}")