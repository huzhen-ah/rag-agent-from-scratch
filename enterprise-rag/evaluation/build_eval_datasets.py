import collections
import datetime
import json
import os
import random
import re


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_corpus_path = os.path.join(
    project_dir,
    "evaluation",
    "source",
    "rag_documents.jsonl",
)
ready_corpus_path = os.path.join(
    project_dir,
    "documents",
    "tenant-001",
    "kb-fault-codes",
    "doc-rag-corpus",
    "v1",
    "rag_corpus.jsonl",
)
output_dir = os.path.join(project_dir, "evaluation", "data")
retrieval_output_path = os.path.join(output_dir, "appliance_retrieval_eval.jsonl")
agent_output_path = os.path.join(output_dir, "appliance_agent_eval.jsonl")
manifest_output_path = os.path.join(output_dir, "manifest.json")


appliance_type_zh = {
    "washer": "洗衣机",
    "dishwasher": "洗碗机",
    "dryer": "烘干机",
    "refrigerator": "冰箱",
    "oven_range": "烤箱/灶具",
}


def load_jsonl(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_content_field(content, field_name):
    match = re.search(r"{}：([^\n]+)".format(re.escape(field_name)), content)
    if match is None:
        return ""
    return match.group(1).strip()


def normalize_text(value):
    value = value.casefold().strip()
    value = re.sub(r"[\s，。；：、,.!?！？;:]", "", value)
    return value


def naturalize_meaning(meaning):
    text = meaning.strip().rstrip("。")
    exact_replacements = {
        "机器未检测到门已关闭": "门关上后仍无法启动",
        "机器无法在预计时间内排水": "一直无法正常排水",
        "机器在预期时间内未排水": "一直无法正常排水",
        "机器无法排水": "一直无法排水",
        "洗碗机无法排水": "一直无法排水",
        "过量洗涤剂（过度起泡）检测到": "洗涤剂过量并产生大量泡沫",
    }
    if text in exact_replacements:
        return exact_replacements[text]
    replacements = (
        ("故障检测到", "出现故障"),
        ("错误检测到", "出现错误"),
        ("检测到", "提示"),
        ("报告故障", "提示异常"),
        ("报故障", "提示异常"),
        ("故障指示", "出现异常"),
        ("无法在预计时间内", "一直无法"),
        ("未完全", "没有完全"),
        ("故障出现", "出现故障"),
        ("错误出现", "出现错误"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def market_zh(market):
    return {"US": "美国版", "UK": "英国版"}.get(market, market)


def add_runtime_fields(processed_rows, ready_rows):
    id_to_chunk_id = {row["id"]: chunk_id for chunk_id, row in enumerate(ready_rows)}
    ready_by_id = {row["id"]: row for row in ready_rows}
    rows = []
    for row in processed_rows:
        if row["id"] not in id_to_chunk_id:
            raise ValueError("ready corpus 缺少文档: {}".format(row["id"]))
        item = dict(row)
        item["chunk_id"] = id_to_chunk_id[row["id"]]
        item["ready_content"] = ready_by_id[row["id"]]["content"]
        item["meaning"] = extract_content_field(item["ready_content"], "故障含义")
        item["appliance_type_zh"] = appliance_type_zh[item["appliance_type"]]
        rows.append(item)
    return rows


def build_indexes(rows):
    code_index = collections.defaultdict(list)
    symptom_index = collections.defaultdict(list)
    exact_index = collections.defaultdict(list)
    for row in rows:
        code_key = (
            row["brand"].casefold(),
            row["appliance_type"],
            row["error_code"].casefold(),
        )
        symptom_key = (
            row["brand"].casefold(),
            row["appliance_type"],
            normalize_text(row["meaning"]),
        )
        exact_key = (
            row["brand"].casefold(),
            row["market"],
            row["appliance_type"],
            row["error_code"].casefold(),
            normalize_text(row["meaning"]),
        )
        code_index[code_key].append(row)
        symptom_index[symptom_key].append(row)
        exact_index[exact_key].append(row)
    return code_index, symptom_index, exact_index


def relevance_fields(rows):
    ordered = sorted(rows, key=lambda row: row["chunk_id"])
    return {
        "relevant_chunk_ids": [row["chunk_id"] for row in ordered],
        "relevant_document_ids": [row["id"] for row in ordered],
    }


def build_retrieval_dataset(test_rows, code_index, symptom_index, exact_index):
    cases = []
    direct_templates = (
        "我的{brand}{appliance}显示{code}，这个代码是什么意思？",
        "{brand}{appliance}报错{code}，应该怎么处理？",
        "{brand}{appliance}显示{code}，是什么意思？",
    )
    symptom_templates = (
        "我的{brand}{appliance}{symptom}，可能是什么问题？",
        "{brand}{appliance}现在{symptom}，应该怎么排查？",
        "家里的{brand}{appliance}{symptom}，接下来怎么办？",
    )
    mixed_templates = (
        "{brand}{market}{appliance}报{code}，同时{symptom}，应该怎么处理？",
        "我的{brand}{market}{appliance}出现{code}，表现为{symptom}，请给出排查建议。",
        "{brand}{appliance}显示{code}并且{symptom}，这是什么故障？",
    )

    for row in sorted(test_rows, key=lambda item: item["chunk_id"]):
        template_index = row["chunk_id"] % 3
        values = {
            "brand": row["brand"],
            "market": market_zh(row["market"]),
            "appliance": row["appliance_type_zh"],
            "code": row["error_code"],
            "symptom": naturalize_meaning(row["meaning"]),
        }

        code_key = (
            row["brand"].casefold(),
            row["appliance_type"],
            row["error_code"].casefold(),
        )
        symptom_key = (
            row["brand"].casefold(),
            row["appliance_type"],
            normalize_text(row["meaning"]),
        )
        exact_key = (
            row["brand"].casefold(),
            row["market"],
            row["appliance_type"],
            row["error_code"].casefold(),
            normalize_text(row["meaning"]),
        )

        variants = (
            (
                "code_lookup",
                direct_templates[template_index].format(**values),
                code_index[code_key],
            ),
            (
                "symptom_lookup",
                symptom_templates[template_index].format(**values),
                symptom_index[symptom_key],
            ),
            (
                "code_and_symptom",
                mixed_templates[template_index].format(**values),
                exact_index[exact_key],
            ),
        )

        for query_type, question, relevant_rows in variants:
            case = {
                "question": question,
                "query_type": query_type,
                "relevant_record_ids": relevance_fields(relevant_rows)[
                    "relevant_document_ids"
                ],
            }
            cases.append(case)
    for retrieve_id, case in enumerate(cases, start=1):
        case["retrieve_id"] = retrieve_id
    return cases


def balanced_order(rows, seed):
    random_generator = random.Random(seed)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["appliance_type"]].append(row)
    for values in grouped.values():
        random_generator.shuffle(values)

    ordered = []
    type_order = ["washer", "dishwasher", "dryer", "refrigerator", "oven_range"]
    while any(grouped.values()):
        for appliance_type in type_order:
            if grouped[appliance_type]:
                ordered.append(grouped[appliance_type].pop())
    return ordered


def cyclic_take(rows, start, count):
    return [rows[(start + index) % len(rows)] for index in range(count)]


def reference_answer(row):
    repairs = row.get("repairs") or []
    professional_difficulties = {"advanced", "professional_only"}
    return {
        "meaning": row["meaning"],
        "repair_titles": [repair["title_zh"] for repair in repairs],
        "requires_safety_warning": (
            "仅限专业维修人员处理" in row["ready_content"]
            or "不建议普通用户自行处理" in row["ready_content"]
            or any(
                repair.get("difficulty") in professional_difficulties
                for repair in repairs
            )
            or row["appliance_type"] == "oven_range"
        ),
    }


def expected_tool_call(row, code_index, extra_required_facts=None, forbidden_facts=None):
    code_key = (
        row["brand"].casefold(),
        row["appliance_type"],
        row["error_code"].casefold(),
    )
    required_query_facts = [
        row["brand"],
        row["appliance_type_zh"],
        row["error_code"],
    ]
    if extra_required_facts:
        required_query_facts.extend(extra_required_facts)
    relevant = relevance_fields(code_index[code_key])
    expected = {
        "action": "tool_call",
        "tool_name": "query_rag",
        "required_query_facts": required_query_facts,
        "forbidden_query_facts": forbidden_facts or [],
        "reference_answer": reference_answer(row),
    }
    expected.update(relevant)
    return expected


def clarification_expected(missing_fields, required_term_groups):
    return {
        "action": "clarify",
        "must_not_call_tool": True,
        "missing_fields": missing_fields,
        "required_clarification_term_groups": required_term_groups,
    }


def make_case(case_number, category, turns, source_rows):
    return {
        "case_id": "agent_{:04d}".format(case_number),
        "category": category,
        "turns": turns,
        "source_split": "test",
        "source_document_ids": [row["id"] for row in source_rows],
    }


def build_agent_dataset(test_rows, code_index):
    unique_rows = []
    for row in test_rows:
        code_key = (
            row["brand"].casefold(),
            row["appliance_type"],
            row["error_code"].casefold(),
        )
        if len(code_index[code_key]) == 1:
            unique_rows.append(row)

    ordered = balanced_order(unique_rows, seed=20260917)
    cases = []
    case_number = 1

    for row in cyclic_take(ordered, 0, 24):
        user_text = "我的{}{}显示{}，是什么意思，应该怎么处理？".format(
            row["brand"], row["appliance_type_zh"], row["error_code"]
        )
        cases.append(
            make_case(
                case_number,
                "complete_single_turn",
                [{"user": user_text, "expected": expected_tool_call(row, code_index)}],
                [row],
            )
        )
        case_number += 1

    for row in cyclic_take(ordered, 24, 12):
        user_text = "我的{}显示{}，应该怎么办？".format(
            row["appliance_type_zh"], row["error_code"]
        )
        cases.append(
            make_case(
                case_number,
                "missing_brand",
                [
                    {
                        "user": user_text,
                        "expected": clarification_expected(["brand"], [["品牌"]]),
                    }
                ],
                [row],
            )
        )
        case_number += 1

    for row in cyclic_take(ordered, 36, 12):
        user_text = "我的{}设备显示{}，应该怎么办？".format(
            row["brand"], row["error_code"]
        )
        cases.append(
            make_case(
                case_number,
                "missing_appliance_type",
                [
                    {
                        "user": user_text,
                        "expected": clarification_expected(
                            ["appliance_type"],
                            [["家电类型", "设备类型", "什么设备"]],
                        ),
                    }
                ],
                [row],
            )
        )
        case_number += 1

    for row in cyclic_take(ordered, 48, 12):
        user_text = "我的{}{}出问题了，应该怎么办？".format(
            row["brand"], row["appliance_type_zh"]
        )
        cases.append(
            make_case(
                case_number,
                "missing_fault_detail",
                [
                    {
                        "user": user_text,
                        "expected": clarification_expected(
                            ["error_code_or_symptom"],
                            [["故障代码", "故障现象", "异常现象"]],
                        ),
                    }
                ],
                [row],
            )
        )
        case_number += 1

    for row in cyclic_take(ordered, 60, 12):
        turns = [
            {
                "user": "我家的{}报错了。".format(row["appliance_type_zh"]),
                "expected": clarification_expected(
                    ["brand", "error_code_or_symptom"],
                    [["品牌"], ["故障代码", "故障现象", "异常现象"]],
                ),
            },
            {
                "user": "品牌是{}，完整代码是{}。".format(row["brand"], row["error_code"]),
                "expected": expected_tool_call(row, code_index),
            },
        ]
        cases.append(make_case(case_number, "multi_turn_completion", turns, [row]))
        case_number += 1

    correction_targets = cyclic_take(ordered, 72, 12)
    same_type_rows = collections.defaultdict(list)
    for row in ordered:
        same_type_rows[row["appliance_type"]].append(row)
    for correction_index, row in enumerate(correction_targets):
        old_candidates = [
            candidate
            for candidate in same_type_rows[row["appliance_type"]]
            if candidate["brand"] != row["brand"]
            and candidate["error_code"].casefold() != row["error_code"].casefold()
        ]
        if not old_candidates:
            old_candidates = [
                candidate
                for candidate in same_type_rows[row["appliance_type"]]
                if candidate["brand"] != row["brand"]
            ]
        old_row = old_candidates[correction_index % len(old_candidates)]
        turns = [
            {
                "user": "我的{}{}显示{}，应该怎么处理？".format(
                    old_row["brand"],
                    old_row["appliance_type_zh"],
                    old_row["error_code"],
                ),
                "expected": expected_tool_call(old_row, code_index),
            },
            {
                "user": "刚才品牌和代码都说错了，实际是{}，代码是{}。".format(
                    row["brand"], row["error_code"]
                ),
                "expected": expected_tool_call(
                    row,
                    code_index,
                    forbidden_facts=(
                        [old_row["brand"]]
                        + (
                            [old_row["error_code"]]
                            if old_row["error_code"].casefold()
                            != row["error_code"].casefold()
                            else []
                        )
                    ),
                ),
            },
        ]
        cases.append(
            make_case(case_number, "correction_override", turns, [old_row, row])
        )
        case_number += 1

    repair_rows = balanced_order(
        [row for row in test_rows if row.get("repairs")], seed=20260918
    )
    for row in cyclic_take(repair_rows, 0, 12):
        first_repair = row["repairs"][0]["title_zh"]
        if len(row["repairs"]) > 1:
            required_answer_facts = [row["repairs"][1]["title_zh"]]
        else:
            required_answer_facts = ["专业维修"]
        turns = [
            {
                "user": "我的{}{}显示{}，怎么处理？".format(
                    row["brand"], row["appliance_type_zh"], row["error_code"]
                ),
                "expected": expected_tool_call(row, code_index),
            },
            {
                "user": "我已经{}了，还是没有解决，下一步怎么办？".format(first_repair),
                "expected": {
                    "action": "answer_from_history",
                    "must_not_call_tool": True,
                    "required_context_facts": [
                        row["brand"],
                        row["appliance_type_zh"],
                        row["error_code"],
                        first_repair,
                    ],
                    "required_answer_facts": required_answer_facts,
                    "reference_answer": reference_answer(row),
                },
            },
        ]
        cases.append(
            make_case(case_number, "followup_with_prior_evidence", turns, [row])
        )
        case_number += 1

    return cases


def validate_datasets(retrieval_cases, agent_cases, ready_rows, rows):
    valid_record_ids = {row["id"] for row in ready_rows}
    test_document_ids = {
        row["id"]
        for row in rows
        if row["split"] == "test"
    }
    agent_ids = [case["case_id"] for case in agent_cases]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("Agent 评测 case_id 重复")
    if len(retrieval_cases) != 159:
        raise ValueError("检索评测样本应为159条，实际为{}条".format(len(retrieval_cases)))
    if len(agent_cases) != 96:
        raise ValueError("Agent评测场景应为96条，实际为{}条".format(len(agent_cases)))
    if len({case["question"] for case in retrieval_cases}) != len(retrieval_cases):
        raise ValueError("检索评测问题重复")

    expected_retrieval_type_counts = {
        "code_lookup": 53,
        "symptom_lookup": 53,
        "code_and_symptom": 53,
    }
    actual_retrieval_type_counts = collections.Counter(
        case["query_type"] for case in retrieval_cases
    )
    if actual_retrieval_type_counts != expected_retrieval_type_counts:
        raise ValueError(
            "检索类型分布错误: {}".format(dict(actual_retrieval_type_counts))
        )

    expected_agent_category_counts = {
        "complete_single_turn": 24,
        "missing_brand": 12,
        "missing_appliance_type": 12,
        "missing_fault_detail": 12,
        "multi_turn_completion": 12,
        "correction_override": 12,
        "followup_with_prior_evidence": 12,
    }
    actual_agent_category_counts = collections.Counter(
        case["category"] for case in agent_cases
    )
    if actual_agent_category_counts != expected_agent_category_counts:
        raise ValueError(
            "Agent类别分布错误: {}".format(dict(actual_agent_category_counts))
        )

    for case in retrieval_cases:
        relevant_ids = case["relevant_record_ids"]
        if not relevant_ids:
            raise ValueError("检索样本没有相关文档: {}".format(case["question"]))
        if not set(relevant_ids).issubset(valid_record_ids):
            raise ValueError("检索样本含非法record_id: {}".format(case["question"]))

    for case in agent_cases:
        if case["source_split"] != "test":
            raise ValueError("Agent场景不是test来源: {}".format(case["case_id"]))
        if not set(case["source_document_ids"]).issubset(test_document_ids):
            raise ValueError("Agent场景引用了非test源文档: {}".format(case["case_id"]))
        if not case["turns"]:
            raise ValueError("Agent场景没有turn: {}".format(case["case_id"]))
        for turn in case["turns"]:
            if turn["expected"]["action"] == "tool_call":
                if turn["expected"]["tool_name"] != "query_rag":
                    raise ValueError("Agent工具名错误: {}".format(case["case_id"]))
                if len(turn["expected"]["required_query_facts"]) < 3:
                    raise ValueError("Agent工具参数事实不足: {}".format(case["case_id"]))
                required_facts = {
                    normalize_text(value)
                    for value in turn["expected"]["required_query_facts"]
                }
                forbidden_facts = {
                    normalize_text(value)
                    for value in turn["expected"]["forbidden_query_facts"]
                }
                if required_facts & forbidden_facts:
                    raise ValueError(
                        "Agent工具参数必需事实与禁止事实冲突: {}".format(
                            case["case_id"]
                        )
                    )


def build_manifest(retrieval_cases, agent_cases, test_rows):
    retrieval_types = collections.Counter(case["query_type"] for case in retrieval_cases)
    agent_categories = collections.Counter(case["category"] for case in agent_cases)
    agent_turns = sum(len(case["turns"]) for case in agent_cases)
    return {
        "generated_at": datetime.date.today().isoformat(),
        "source": "evaluation/source/rag_documents.jsonl",
        "source_split": "test",
        "source_documents": len(test_rows),
        "retrieval_cases": len(retrieval_cases),
        "retrieval_query_type_counts": dict(sorted(retrieval_types.items())),
        "agent_scenarios": len(agent_cases),
        "agent_turns": agent_turns,
        "agent_category_counts": dict(sorted(agent_categories.items())),
        "brands": sorted({row["brand"] for row in test_rows}),
        "appliance_types": sorted({row["appliance_type"] for row in test_rows}),
        "notes": [
            "评测目标文档全部来自原始数据的test划分。",
            "检索问题由故障代码、故障含义和元数据按固定模板生成，未调用待评测模型。",
            "Agent评测同时覆盖工具调用、缺参追问、多轮补全、信息纠正和历史证据复用。",
        ],
    }


def main():
    processed_rows = load_jsonl(processed_corpus_path)
    ready_rows = load_jsonl(ready_corpus_path)
    rows = add_runtime_fields(processed_rows, ready_rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    code_index, symptom_index, exact_index = build_indexes(rows)

    retrieval_cases = build_retrieval_dataset(
        test_rows, code_index, symptom_index, exact_index
    )
    agent_cases = build_agent_dataset(test_rows, code_index)
    validate_datasets(retrieval_cases, agent_cases, ready_rows, rows)

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(retrieval_output_path, retrieval_cases)
    write_jsonl(agent_output_path, agent_cases)
    manifest = build_manifest(retrieval_cases, agent_cases, test_rows)
    with open(manifest_output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(retrieval_output_path)
    print(agent_output_path)
    print(manifest_output_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
