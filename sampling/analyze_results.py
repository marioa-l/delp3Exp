import argparse
import json
import os
import sys


def compute_metric(approximate, exact):
    """
    Compute the quality metric based on the remainder of the interval.
    Metric is defined as remainder_approximate / remainder_exact.
    A metric closer to 1 is better (means it approaches the exact interval).
    If exact width is 1 (meaning remainder is 0), returns 0 to avoid division by zero.
    """
    width_approximate = approximate[1] - approximate[0]
    width_exact = exact[1] - exact[0]
    remainder_approximate = 1 - width_approximate
    remainder_exact = 1 - width_exact

    if remainder_exact == 0:
        return 0.0
    return remainder_approximate / remainder_exact


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_report(exact_path, worlds_path, progs_path, report_out):
    try:
        exact_data = load_json(exact_path)
        worlds_data = load_json(worlds_path)
        progs_data = load_json(progs_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    literals = list(exact_data["status"].keys())

    report_lines = []
    report_lines.append("# Literal Approximation Quality Report\n")
    report_lines.append(
        f"- **Worlds Approach**: {worlds_data['data']['delp_calls']} DeLP calls, {worlds_data['data']['time']:.2f}s total time"
    )
    report_lines.append(
        f"- **Programs Approach**: {progs_data['data']['delp_calls']} DeLP calls, {progs_data['data']['time']:.2f}s total time\n"
    )

    report_lines.append(
        "| Literal | Exact [L, U] | Worlds [L, U] | Worlds Metric | Progs [L, U] | Progs Metric | Recommended |"
    )
    report_lines.append(
        "|---------|--------------|---------------|---------------|--------------|--------------|-------------|"
    )

    for lit in sorted(literals):
        # Extract Exact
        exact_status = exact_data["status"].get(lit)
        if not exact_status:
            continue
        ex_l, ex_u = (
            exact_status.get("l", exact_status.get("pyes", 0)),
            exact_status.get("u", 1 - exact_status.get("pno", 0)),
        )

        # Extract Worlds
        w_status = worlds_data["status"].get(lit, {})
        w_l, w_u = (
            w_status.get("l", w_status.get("pyes", 0)),
            w_status.get("u", 1 - w_status.get("pno", 0)),
        )
        w_metric = compute_metric([w_l, w_u], [ex_l, ex_u])

        # Extract Programs
        p_status = progs_data["status"].get(lit, {})
        p_l, p_u = (
            p_status.get("l", p_status.get("pyes", 0)),
            p_status.get("u", 1 - p_status.get("pno", 0)),
        )
        p_metric = compute_metric([p_l, p_u], [ex_l, ex_u])

        # Recommendation
        # The metric closer to 1.0 (from below, as it's an approximation) is better.
        # But we also might want to check the error |1.0 - metric|
        err_w = abs(1.0 - w_metric)
        err_p = abs(1.0 - p_metric)

        if err_w < err_p:
            recommended = "Worlds"
        elif err_p < err_w:
            recommended = "Programs"
        else:
            recommended = "Tie"

        report_lines.append(
            f"| `{lit}` | [{ex_l:.4f}, {ex_u:.4f}] | [{w_l:.4f}, {w_u:.4f}] | {w_metric:.4f} | [{p_l:.4f}, {p_u:.4f}] | {p_metric:.4f} | **{recommended}** |"
        )

    report_content = "\n".join(report_lines)

    with open(report_out, "w") as f:
        f.write(report_content)

    print(f"Report successfully saved to {report_out}")
    print("\n--- Summary of Report ---")
    print(report_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze approximation methods and recommend the best per literal."
    )
    parser.add_argument("--exact", help="Path to exact JSON", required=True)
    parser.add_argument("--worlds", help="Path to Worlds sampling JSON", required=True)
    parser.add_argument("--progs", help="Path to Programs sampling JSON", required=True)
    parser.add_argument("--out", help="Path to save the report", required=True)
    args = parser.parse_args()

    generate_report(args.exact, args.worlds, args.progs, args.out)
