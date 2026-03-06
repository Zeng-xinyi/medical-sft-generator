import json
import random
import re

# -------------------------
# 配置
# -------------------------

INPUT_PATH = "patients.json"
OUTPUT_PATH = "seed.json"

# 医学能力类型
competencies = [
    "Symptom Gathering",
    "Differential Diagnosis",
    "Recommend Tests",
    'Interpretation',
    'Final Diagnosis',
    'Treatment Plan'
]

# 难度等级
difficulties = [
    "easy",
    "medium",
    "hard"
]

# 科室中英文映射（可扩展）
department_map = {
    "内科": "Internal Medicine",
    "儿科": "Pediatrics",
    "妇产科": "Obstetrics and Gynecology",
    "耳鼻咽喉科": "Otorhinolaryngology",
    "外科": "Surgery",
}

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

import itertools

seeds = []

# 所有组合
combos = list(itertools.product(competencies, difficulties))

# 每个 department 的计数
dept_counter = {}

for case in cases:

    department_cn = case.get("department", "")

    department_en = None

    # -------------------------
    # 模糊匹配 department
    # -------------------------
    for key in department_map:
        if key in department_cn:
            department_en = department_map[key]
            break

    if department_en is None:
        continue

    # 初始化计数
    if department_en not in dept_counter:
        dept_counter[department_en] = 0

    # 每个科室最多15条
    if dept_counter[department_en] >= 15:
        continue

    # -------------------------
    # 获取疾病
    # -------------------------
    diseases = case.get("diseases")

    if diseases:
        disease = diseases
    else:
        disease = None

    # -------------------------
    # 选择组合
    # -------------------------
    combo_id = dept_counter[department_en]

    competency, difficulty = combos[combo_id]

    seed = {
        "id": len(seeds) + 1,
        "department": department_en,
        "competency": competency,
        "difficulty": difficulty,
        "disease_example": disease
    }

    seeds.append(seed)

    dept_counter[department_en] += 1
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(seeds, f, indent=2, ensure_ascii=False)

print(f"Generated {len(seeds)} seeds.")
print("Saved to data/seed.json")