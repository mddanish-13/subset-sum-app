import io
import math
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

# --- Try PuLP first (preferred, matches your original ILP idea) ---
pulp_available = True
try:
    from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize, PULP_CBC_CMD, LpBinary
except Exception:
    pulp_available = False

# --- OR-Tools fallback (robust on Streamlit Cloud) ---
ortools_available = True
try:
    from ortools.sat.python import cp_model
except Exception:
    ortools_available = False


st.set_page_config(page_title="Subset Sum Finder", layout="wide")
st.title("Subset Sum Finder (ILP • Multi-Combination)")
st.caption("Upload numbers (CSV/Excel) or paste them. Get one or more subsets that sum to a target. "
           "Best combo = fewest items. Fallback solver ensures it runs on the cloud.")


# ----------------------------- helpers -----------------------------
def clean_numbers(seq):
    out = []
    for x in seq:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def scale_to_ints(numbers, target, max_decimals=6):
    """
    Convert floats to integers by scaling (so MILP/CP-SAT can use equality exactly).
    """
    if len(numbers) == 0:
        return [], 0, 1

    # Detect max decimals
    def decimals(x):
        s = f"{x:.{max_decimals}f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
        if "." in s:
            return len(s.split(".")[1])
        return 0

    d = 0
    for v in numbers + [target]:
        d = max(d, decimals(float(v)))

    scale = 10 ** min(d, max_decimals)
    scaled_nums = [int(round(v * scale)) for v in numbers]
    scaled_target = int(round(float(target) * scale))
    return scaled_nums, scaled_target, scale


def combinations_via_pulp(numbers, target, k=1):
    """
    Exact ILP using PuLP. Minimizes number of items to give 'best' first.
    Finds k distinct combinations by adding 'no-good' cuts iteratively.
    """
    results = []
    scaled_nums, scaled_target, scale = scale_to_ints(numbers, target)

    # Build once, then clone with new constraints to iterate
    n = len(scaled_nums)
    x_vars = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(n)]
    base_prob = LpProblem("SubsetSum", LpMinimize)
    # Objective: minimize number of items (best = fewest elements)
    base_prob += lpSum(x_vars)
    # Sum constraint
    base_prob += lpSum([scaled_nums[i] * x_vars[i] for i in range(n)]) == scaled_target

    # Use CBC if available; suppress logs
    solver = PULP_CBC_CMD(msg=False)

    # We iterate by adding "no-good cuts"
    used_cuts = []
    # defensive bound
    max_iters = max(1, int(k))

    for _ in range(max_iters):
        prob = base_prob.copy()
        # Add all previous no-good cuts
        for sel in used_cuts:
            idxs = [i for i, v in enumerate(sel) if v == 1]
            prob += lpSum([x_vars[i] for i in idxs]) <= (len(idxs) - 1)

        status = prob.solve(solver)
        if LpStatus.get(status, "Unknown") != "Optimal":
            break

        sol = [int(round(v.value())) for v in x_vars]
        if sum(sol) == 0:
            break
        combo = [numbers[i] for i, bit in enumerate(sol) if bit == 1]
        results.append(combo)
        used_cuts.append(sol)

    return results


def combinations_via_ortools(numbers, target, k=1, time_limit_sec=10):
    """
    Exact ILP using OR-Tools CP-SAT. Minimizes number of items.
    Finds k distinct combinations by iterative 'no-good' cuts.
    """
    results = []
    scaled_nums, scaled_target, scale = scale_to_ints(numbers, target)
    n = len(scaled_nums)

    # We'll iterate: each time forbid the previous pattern
    used_patterns = []
    for _ in range(int(k)):
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Sum constraint
        model.Add(sum(scaled_nums[i] * x[i] for i in range(n)) == scaled_target)
        # Objective: minimize number of items
        model.Minimize(sum(x))

        # No-good cuts
        for pat in used_patterns:
            idxs = [i for i, b in enumerate(pat) if b == 1]
            if idxs:
                model.Add(sum(x[i] for i in idxs) <= len(idxs) - 1)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_sec
        solver.parameters.num_search_workers = 8

        res = solver.Solve(model)
        if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
            break

        sel = [int(solver.Value(x[i])) for i in range(n)]
        if sum(sel) == 0:
            break
        combo = [numbers[i] for i, bit in enumerate(sel) if bit == 1]
        results.append(combo)
        used_patterns.append(sel)

    return results


def find_k_combinations(numbers, target, k=1):
    """
    Try PuLP first (to honor the original approach). If unavailable or fails, use OR-Tools.
    """
    start = time.time()
    solver_used = None
    combos = []

    if pulp_available:
        try:
            combos = combinations_via_pulp(numbers, target, k=k)
            solver_used = "PuLP (CBC)"
        except Exception as e:
            solver_used = f"PuLP failed: {e}"

    if not combos and ortools_available:
        combos = combinations_via_ortools(numbers, target, k=k)
        solver_used = "OR-Tools CP-SAT"

    elapsed = time.time() - start
    return combos, solver_used or "No solver available", elapsed


def results_to_tables(combos):
    """
    Produce two tables:
      1) Long format: one row per (combo_id, item_idx) with values
      2) Wide format: one row per combination, items spread across columns
    """
    if not combos:
        return None, None

    long_rows = []
    for cid, combo in enumerate(combos, start=1):
        for idx, val in enumerate(combo, start=1):
            long_rows.append({"Combination_ID": cid, "Item_Index": idx, "Value": val})
    df_long = pd.DataFrame(long_rows)

    max_len = max(len(c) for c in combos)
    wide_cols = [f"Item_{i}" for i in range(1, max_len + 1)]
    data = []
    for cid, combo in enumerate(combos, start=1):
        row = combo + [np.nan] * (max_len - len(combo))
        data.append([cid] + row + [sum(combo), len(combo)])
    df_wide = pd.DataFrame(data, columns=["Combination_ID"] + wide_cols + ["Sum", "Count"])

    return df_long, df_wide


def make_excel_download(df_long, df_wide):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_wide.to_excel(writer, index=False, sheet_name="Combinations")
        if df_long is not None:
            df_long.to_excel(writer, index=False, sheet_name="Long_Format")
    return output.getvalue()


# ----------------------------- UI -----------------------------
with st.sidebar:
    st.header("Input")
    src = st.radio("Choose input method", ["Upload CSV/Excel", "Paste comma-separated"])
    k = st.number_input("How many combinations do you want?", min_value=1, max_value=20, value=2, step=1,
                        help="We return the 'best' (fewest items) first, then additional distinct combos.")
    st.info("Tip: Integers are fastest. Decimals are supported (we scale internally).", icon="💡")

numbers = []

if src == "Upload CSV/Excel":
    up = st.file_uploader("Upload a CSV or Excel file (first column = numbers)", type=["csv", "xlsx"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.read_excel(up)
        col = df.columns[0]
        numbers = clean_numbers(df[col].tolist())
        st.write("Preview:", df.head())
else:
    txt = st.text_area("Paste numbers separated by commas", "2,3,7,8,10")
    if txt.strip():
        numbers = clean_numbers([x.strip() for x in txt.split(",")])

c1, c2 = st.columns(2)
with c1:
    target = st.number_input("Target sum", value=10.0, step=1.0)
with c2:
    st.write("")  # spacing
    run = st.button("Find combinations", type="primary")

if run:
    if not numbers:
        st.error("Please provide some numbers.")
    else:
        with st.spinner("Solving…"):
            combos, solver_used, seconds = find_k_combinations(numbers, target, k=int(k))

        st.subheader("Results")
        st.caption(f"Solver: **{solver_used}** | Time: **{seconds:.3f} s**")
        if combos:
            # Best first (fewest items)
            combos = sorted(combos, key=lambda c: (len(c), -sum(c)))
            for i, c in enumerate(combos, 1):
                st.write(f"**Combination {i}** (count={len(c)}, sum={sum(c)}): {c}")

            df_long, df_wide = results_to_tables(combos)
            st.dataframe(df_wide, use_container_width=True)

            csv_bytes = df_wide.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, file_name="subset_results.csv", mime="text/csv")

            xlsx_bytes = make_excel_download(df_long, df_wide)
            st.download_button("Download Excel", xlsx_bytes, file_name="subset_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.error("No combination found for the given target.")
