
import numpy as np
import re

""" 
Source: Processing functions from Wikisql dataset used for calculating test metrics
"""

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]
COND_OPS = ["=", ">", "<"]

def computeLFEM(greedy, answer):
    count = 0
    correct = 0
    text_answers = []
    for idx, (g, ex) in enumerate(zip(greedy, answer)):
        count += 1
        text_answers.append([ex["answer"].lower()])
        try:
            gt = ex["sql"]
            conds = gt["conds"]
            lower_conds = []
            for c in conds:
                lc = c
                lc[2] = str(lc[2]).lower().replace(" ", "")
                lower_conds.append(lc)
            gt["conds"] = lower_conds
            lf = to_lf(g, ex["table"])
            correct += lf == gt
        except Exception as e:
            continue
    return (correct / count) * 100, text_answers

def to_lf(s, table):
    aggs = [y.lower() for y in AGG_OPS]
    agg_to_idx = {x: i for i, x in enumerate(aggs)}
    conditionals = [y.lower() for y in COND_OPS]
    headers_unsorted = [(y.lower(), i) for i, y in enumerate(table["header"])]
    headers = [(y.lower(), i) for i, y in enumerate(table["header"])]
    headers.sort(reverse=True, key=lambda x: len(x[0]))
    condition_s, conds = None, []
    if "where" in s:
        s, condition_s = s.split("where", 1)

    s = " ".join(s.split()[1:-2])
    s_no_agg = " ".join(s.split()[1:])
    sel, agg = None, 0
    lcss, idxs = [], []
    for col, idx in headers:
        lcss.append(lcs(col, s))
        lcss.append(lcs(col, s_no_agg))
        idxs.append(idx)
    lcss = np.array(lcss)
    max_id = np.argmax(lcss)
    sel = idxs[max_id // 2]
    if max_id % 2 == 1:  # with agg
        agg = agg_to_idx[s.split()[0]]

    full_conditions = []
    if not condition_s is None:

        pattern = "|".join(COND_OPS)
        split_conds_raw = re.split(pattern, condition_s)
        split_conds_raw = [conds.strip() for conds in split_conds_raw]
        split_conds = [split_conds_raw[0]]
        for i in range(1, len(split_conds_raw) - 1):
            split_conds.extend(re.split("and", split_conds_raw[i]))
        split_conds += [split_conds_raw[-1]]
        for i in range(0, len(split_conds), 2):
            cur_s = split_conds[i]
            lcss = []
            for col in headers:
                lcss.append(lcs(col[0], cur_s))
            max_id = np.argmax(np.array(lcss))
            split_conds[i] = headers[max_id][0]
        for i, m in enumerate(re.finditer(pattern, condition_s)):
            split_conds[2 * i] = split_conds[2 * i] + " " + m.group()
        split_conds = [
            " ".join(split_conds[2 * i : 2 * i + 2])
            for i in range(len(split_conds) // 2)
        ]

        condition_s = " and ".join(split_conds)
        condition_s = " " + condition_s + " "
        for idx, col in enumerate(headers):
            condition_s = condition_s.replace(
                " " + col[0] + " ", " Col{} ".format(col[1])
            )
        condition_s = condition_s.strip()

        for idx, col in enumerate(conditionals):
            new_s = []
            for t in condition_s.split():
                if t == col:
                    new_s.append("Cond{}".format(idx))
                else:
                    new_s.append(t)
            condition_s = " ".join(new_s)
        s = condition_s

        conds = re.split("(Col\d+ Cond\d+)", s)
        if len(conds) == 0:
            conds = [s]
        conds = [x for x in conds if len(x.strip()) > 0]
        full_conditions = []
        for i, x in enumerate(conds):
            if i % 2 == 0:
                x = x.split()
                col_num = int(x[0].replace("Col", ""))
                opp_num = int(x[1].replace("Cond", ""))
                full_conditions.append([col_num, opp_num])
            else:
                x = x.split()
                if x[-1] == "and":
                    x = x[:-1]
                x = " ".join(x)
                if "Col" in x:
                    new_x = []
                    for t in x.split():
                        if "Col" in t:
                            idx = int(t.replace("Col", ""))
                            t = headers_unsorted[idx][0]
                        new_x.append(t)
                    x = new_x
                    x = " ".join(x)
                if "Cond" in x:
                    new_x = []
                    for t in x.split():
                        if "Cond" in t:
                            idx = int(t.replace("Cond", ""))
                            t = conditionals[idx]
                        new_x.append(t)
                    x = new_x
                    x = " ".join(x)
                full_conditions[-1].append(x.replace(" ", ""))
    logical_form = {"sel": sel, "conds": full_conditions, "agg": agg}
    return logical_form


def lcsstr(string1, string2):
    answer = 0
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = 0
        for j in range(len2):
            if i + j < len1 and string1[i + j] == string2[j]:
                match += 1
                if match > answer:
                    answer = match
            else:
                match = 0
    return answer


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n] / max(m, n) + lcsstr(X, Y) / 1e4 - min(m, n) / 1e8
