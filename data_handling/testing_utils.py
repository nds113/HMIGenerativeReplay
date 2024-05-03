import collections
import re
import string

def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(
        ground_truth_tokens
    )
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(prediction, ground_truth):
    return prediction == ground_truth


def computeF1(outputs, targets):
    return (sum([metric_max_over_ground_truths(f1_score, o, t)for o, t in zip(outputs, targets)])/ len(outputs)* 100)

def dict_cmp(d1, d2):
    def cmp(a, b):
        for k1, v1 in a.items():
            if k1 not in b:
                return False
            else:
                if v1 != b[k1]:
                    return False
        return True

    return cmp(d1, d2) and cmp(d2, d1)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def to_delta_state(line):
    delta_state = {"inform": {}, "request": {}}
    try:
        if line.lower() == "none" or line.strip() == "" or line.strip() == ";":
            return delta_state
        inform, request = [
            [y.strip() for y in x.strip().split(",")] for x in line.split(";")
        ]
        inform_pairs = {}
        for i in inform:
            try:
                k, v = i.split(":")
                inform_pairs[k.strip()] = v.strip()
            except:
                pass
        delta_state = {"inform": inform_pairs, "request": request}
    except:
        pass
    finally:
        return delta_state

def update_state(state, delta):
    for act, slot in delta.items():
        state[act] = slot
    return state

def computeDialogue(greedy, answer):
    examples = []
    for idx, (g, a) in enumerate(zip(greedy, answer)):
        examples.append((a[0], g, a[1], idx))
    # examples.sort()
    turn_request_positives = 0
    turn_goal_positives = 0
    joint_goal_positives = 0
    ldt = None
    for ex in examples:
        if ldt is None or ldt.split("_")[:-1] != ex[0].split("_")[:-1]:
            state, answer_state = {}, {}
            ldt = ex[0]
        delta_state = to_delta_state(ex[1])
        answer_delta_state = to_delta_state(ex[2])
        state = update_state(state, delta_state["inform"])
        answer_state = update_state(answer_state, answer_delta_state["inform"])
        if dict_cmp(state, answer_state):
            joint_goal_positives += 1
        if delta_state["request"] == answer_delta_state["request"]:
            turn_request_positives += 1
        if dict_cmp(delta_state["inform"], answer_delta_state["inform"]):
            turn_goal_positives += 1

    joint_goal_em = joint_goal_positives / len(examples) * 100
    turn_request_em = turn_request_positives / len(examples) * 100
    turn_goal_em = turn_goal_positives / len(examples) * 100
    answer = [(x[-1], x[-2]) for x in examples]
    # answer.sort()
    answer = [[x[1]] for x in answer]
    return joint_goal_em, turn_request_em, turn_goal_em, answer

def computeEM(outputs, targets):
    outs = [
        metric_max_over_ground_truths(exact_match, o, t)
        for o, t in zip(outputs, targets)
    ]
    return sum(outs) / len(outputs) * 100

def calculate_test_score(data):

    greedy = [datum[0] for datum in data]
    answer = [datum[1] for datum in data]

    metric_keys = []
    metric_values = []

    em = computeEM(greedy, answer)
    metric_keys.append("em")
    metric_values.append(em)
    norm_greedy = [normalize_text(g) for g in greedy]
    norm_answer = [[normalize_text(a) for a in ans] for ans in answer]
    nf1 = computeF1(norm_greedy, norm_answer)
    nem = computeEM(norm_greedy, norm_answer)
    metric_keys.extend(["nf1", "nem"])
    metric_values.extend([nf1, nem])

    metric_dict = collections.OrderedDict(list(zip(metric_keys, metric_values)))
    return metric_dict

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))