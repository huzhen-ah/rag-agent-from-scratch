import json
import os
import random
import re


evaluation_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(evaluation_dir)
workspace_dir = os.path.dirname(project_dir)

retrieval_eval_path = os.path.join(
    evaluation_dir, "data", "appliance_retrieval_eval.jsonl"
)
corpus_path = os.path.join(
    workspace_dir,
    "appliance-support-sft",
    "data",
    "ready",
    "rag_corpus.jsonl",
)
output_dir = os.path.join(evaluation_dir, "data", "prompt_optimization")


appliance_type_zh = {
    "washer": "洗衣机",
    "dishwasher": "洗碗机",
    "dryer": "烘干机",
    "refrigerator": "冰箱",
    "oven_range": "烤箱/灶具",
}

# 这些问法只给出了部件级判断或模糊故障描述，不是可直接检索的故障码或
# 用户可观察到的具体现象。标签来自人工复核，不依据待评测模型的输出自动生成。
clarify_symptom_retrieve_ids = {
    5,
    8,
    11,
    14,
    17,
    26,
    41,
    44,
    50,
    62,
    83,
    92,
    119,
    128,
    131,
}


def load_jsonl(path):
    with open(path, "r", encoding="utf8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_meaning(content):
    match = re.search(r"故障含义：([^\n]+)", content)
    if match is None:
        raise ValueError("语料缺少故障含义")
    return match.group(1).strip().rstrip("。")


def build_groups(retrieval_rows, corpus_rows):
    groups = []
    assert len(retrieval_rows) % 3 == 0
    for start in range(0, len(retrieval_rows), 3):
        positive_rows = retrieval_rows[start : start + 3]
        assert [row["query_type"] for row in positive_rows] == [
            "code_lookup",
            "symptom_lookup",
            "code_and_symptom",
        ]
        source_chunk_id = positive_rows[2]["relevant_chunk_ids"][0]
        source = corpus_rows[source_chunk_id]
        appliance = appliance_type_zh[source["appliance_type"]]
        brand = source["brand"]
        code = source["error_code"]
        meaning = extract_meaning(source["content"])

        cases = []
        for row in positive_rows:
            case = {
                    "user": row["question"],
                }
            if row["retrieve_id"] in clarify_symptom_retrieve_ids:
                case.update(
                    {
                        "expected_action": "clarify",
                        "missing_fields": ["error_code_or_specific_symptom"],
                    }
                )
            else:
                case.update(
                    {
                        "expected_action": "tool_call",
                        "expected_tool_name": "query_rag",
                    }
                )
            cases.append(case)

        negative_cases = [
            {
                "user": "我的{}显示{}，是什么意思，应该怎么处理？".format(
                    appliance, code
                ),
                "expected_action": "clarify",
                "missing_fields": ["brand"],
            },
            {
                "user": "我的{}设备显示{}，应该怎么办？".format(brand, code),
                "expected_action": "clarify",
                "missing_fields": ["appliance_type"],
            },
            {
                "user": "我的{}{}出问题了，应该怎么办？".format(brand, appliance),
                "expected_action": "clarify",
                "missing_fields": ["error_code_or_symptom"],
            },
            {
                "user": "我的{}{}，应该怎么处理？".format(appliance, meaning),
                "expected_action": "clarify",
                "missing_fields": ["brand"],
            },
            {
                "user": "我的{}设备{}，应该怎么处理？".format(brand, meaning),
                "expected_action": "clarify",
                "missing_fields": ["appliance_type"],
            },
            {
                "user": "不知道是什么牌子的{}显示{}，该怎么办？".format(
                    appliance, code
                ),
                "expected_action": "clarify",
                "missing_fields": ["brand"],
            },
            {
                "user": "我的{}家电显示{}，这是什么问题？".format(brand, code),
                "expected_action": "clarify",
                "missing_fields": ["appliance_type"],
            },
            {
                "user": "我的{}{}最近不太正常，帮我看看。".format(
                    brand, appliance
                ),
                "expected_action": "clarify",
                "missing_fields": ["error_code_or_symptom"],
            },
            {
                "user": "不清楚品牌，{}{}，应该怎么排查？".format(
                    appliance, meaning
                ),
                "expected_action": "clarify",
                "missing_fields": ["brand"],
            },
            {
                "user": "{}的一台设备{}，应该怎么排查？".format(brand, meaning),
                "expected_action": "clarify",
                "missing_fields": ["appliance_type"],
            },
        ]
        cases.extend(negative_cases)
        groups.append(
            {
                "source_chunk_id": source_chunk_id,
                "source_document_id": source["id"],
                "has_clarify_symptom": any(
                    row["retrieve_id"] in clarify_symptom_retrieve_ids
                    for row in positive_rows
                ),
                "cases": cases,
            }
        )
    return groups


def deduplicate_cases(groups):
    seen_questions = set()
    for group in groups:
        unique_cases = []
        for case in group["cases"]:
            if case["user"] in seen_questions:
                continue
            seen_questions.add(case["user"])
            unique_cases.append(case)
        group["cases"] = unique_cases
    return groups


def split_groups(groups, seed=20260922):
    random_generator = random.Random(seed)
    clarify_groups = [group for group in groups if group["has_clarify_symptom"]]
    tool_groups = [group for group in groups if not group["has_clarify_symptom"]]
    random_generator.shuffle(clarify_groups)
    random_generator.shuffle(tool_groups)
    return {
        "train": clarify_groups[:9] + tool_groups[:24],
        "dev": clarify_groups[9:12] + tool_groups[24:31],
        "test": clarify_groups[12:15] + tool_groups[31:38],
    }


def flatten(split, groups):
    rows = []
    for group in groups:
        for case in group["cases"]:
            rows.append(dict(case))
    return rows


def balance_rows(rows, seed):
    positives = [row for row in rows if row["expected_action"] == "tool_call"]
    negative_groups = {}
    for row in rows:
        if row["expected_action"] != "clarify":
            continue
        group_key = tuple(row["missing_fields"])
        negative_groups.setdefault(group_key, []).append(row)

    random_generator = random.Random(seed)
    for values in negative_groups.values():
        random_generator.shuffle(values)

    selected_negatives = []
    categories = sorted(negative_groups)
    while len(selected_negatives) < len(positives):
        added = False
        for category in categories:
            values = negative_groups[category]
            if values and len(selected_negatives) < len(positives):
                selected_negatives.append(values.pop())
                added = True
        if not added:
            raise ValueError("负样本不足，无法完成正负平衡采样")

    balanced = positives + selected_negatives
    random_generator.shuffle(balanced)
    return balanced


def validate_group_splits(grouped_splits):
    source_ids = {
        split: {group["source_document_id"] for group in groups}
        for split, groups in grouped_splits.items()
    }
    assert source_ids["train"].isdisjoint(source_ids["dev"])
    assert source_ids["train"].isdisjoint(source_ids["test"])
    assert source_ids["dev"].isdisjoint(source_ids["test"])


def validate(splits):
    questions = {
        split: {row["user"] for row in rows} for split, rows in splits.items()
    }
    assert questions["train"].isdisjoint(questions["dev"])
    assert questions["train"].isdisjoint(questions["test"])
    assert questions["dev"].isdisjoint(questions["test"])

    for split, rows in splits.items():
        tool_calls = sum(row["expected_action"] == "tool_call" for row in rows)
        clarifications = sum(row["expected_action"] == "clarify" for row in rows)
        assert tool_calls > 0
        assert clarifications == tool_calls


def main():
    retrieval_rows = load_jsonl(retrieval_eval_path)
    corpus_rows = load_jsonl(corpus_path)
    groups = build_groups(retrieval_rows, corpus_rows)
    assert len(groups) == 53
    groups = deduplicate_cases(groups)

    grouped_splits = split_groups(groups)
    validate_group_splits(grouped_splits)
    candidate_splits = {
        split: flatten(split, split_groups)
        for split, split_groups in grouped_splits.items()
    }
    negative_pool = [
        row
        for rows in candidate_splits.values()
        for row in rows
        if row["expected_action"] == "clarify"
    ]
    splits = {
        split: balance_rows(rows, seed=20260922 + index)
        for index, (split, rows) in enumerate(candidate_splits.items())
    }
    validate(splits)

    os.makedirs(output_dir, exist_ok=True)
    for split, rows in splits.items():
        output_path = os.path.join(output_dir, "{}.jsonl".format(split))
        write_jsonl(output_path, rows)
        print("{}: {}".format(output_path, len(rows)))
    negative_pool_path = os.path.join(output_dir, "negative_pool.jsonl")
    write_jsonl(negative_pool_path, negative_pool)
    print("{}: {}".format(negative_pool_path, len(negative_pool)))


if __name__ == "__main__":
    main()
