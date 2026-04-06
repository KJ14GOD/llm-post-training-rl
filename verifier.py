import re


START_TAG = "<final_response>"
END_TAG = "</final_response>"

_FINAL_LINE_PATTERN = re.compile(
    rf"^{re.escape(START_TAG)}\s*([+-]?\d+)\s*{re.escape(END_TAG)}$"
)
_TAGGED_PATTERN = re.compile(
    rf"{re.escape(START_TAG)}\s*([+-]?\d+)\s*{re.escape(END_TAG)}",
    re.DOTALL,
)
_INTEGER_PATTERN = re.compile(r"[+-]?\d+")


def trim_to_final_response(text: str) -> str:
    stripped = text.strip()
    end_idx = stripped.find(END_TAG)
    if end_idx == -1:
        return stripped
    return stripped[: end_idx + len(END_TAG)].strip()


def is_exact_format(text: str) -> bool:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    return (
        _FINAL_LINE_PATTERN.fullmatch(lines[-1]) is not None
        and stripped == trim_to_final_response(stripped)
    )


def parse_answer(text: str) -> int | None:
    stripped = trim_to_final_response(text)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    if lines:
        final_line_match = _FINAL_LINE_PATTERN.fullmatch(lines[-1])
        if final_line_match is not None:
            return int(final_line_match.group(1))

    tagged_match = _TAGGED_PATTERN.search(stripped)
    if tagged_match is not None:
        return int(tagged_match.group(1))

    integers = _INTEGER_PATTERN.findall(stripped)
    if integers:
        return int(integers[-1])

    return None


def check_intermediate_steps(raw_text: str, intermediate_results: list[int] | None) -> float:
    """Give partial credit if the model's reasoning contains correct intermediate results."""
    if not intermediate_results:
        return 0.0
    bonus = 0.0
    for result in intermediate_results:
        if str(result) in raw_text:
            bonus += 0.1
    return min(bonus, 0.2)  # cap at 0.2


def reward(parsed_answer: int | None, ground_truth: int, raw_text: str, num_tokens: int, intermediate_results: list[int] | None = None) -> float:
    exact_format = is_exact_format(raw_text)

    if parsed_answer is None:
        score = -0.6
    elif parsed_answer == ground_truth:
        score = 1.0 if exact_format else 0.9
    else:
        abs_error = abs(parsed_answer - ground_truth)
        scale = max(1, abs(ground_truth))
        relative_error = abs_error / scale

        if abs_error == 1:
            score = -0.05
        elif relative_error <= 0.05:
            score = -0.10
        elif relative_error <= 0.20:
            score = -0.15
        else:
            score = -0.25

        # partial credit for correct intermediate steps in reasoning
        score += check_intermediate_steps(raw_text, intermediate_results)

    if num_tokens > 128:
        score -= 0.05

    return score
