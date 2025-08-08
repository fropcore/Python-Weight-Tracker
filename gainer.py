#!/usr/bin/env python3
"""
Weight Gain Tracker — pro-gain terminal stats + shareable HTML
Features
- Reads CSV with REQUIRED start and goal rows
- Computes BMI, pace/week, and progress to goal (kg/lb and %)
- Terminal summary + progress bar
- HTML report (dark, offline) with green=up, red=down, and a goal line

CSV (required keys first, then data):
start_weight,70,kg
goal_weight,85,kg
date,weight,unit
2025-07-01,70,kg
2025-07-08,71,kg
2025-07-15,72,kg

Usage:
  python3 gainer.py --csv sample/sample_weights.csv --height 180cm --name "You" --out report.html
  python3 gainer.py --csv sample/sample_weights.csv --height "5'11\"" --name "You"

No external dependencies.
"""
import csv, sys, re
from datetime import datetime
from typing import List, Tuple, Optional

# ----------------- parsing helpers -----------------
def parse_height(h: str) -> float:
    hs = h.strip().lower().replace(" ", "")
    m = re.match(r'^(\d+(\.\d+)?)m?$', hs)
    if m and 'cm' not in hs and "'" not in hs and 'ft' not in hs:
        val = float(m.group(1))
        if val > 3.0:  # probably cm typed without unit
            return val / 100.0
        return val
    m = re.match(r'^(\d+(\.\d+)?)cm$', hs)
    if m:
        return float(m.group(1)) / 100.0
    m = re.match(r'^(\d+)(\'|ft)?(\d+)?(\"|in)?$', hs)  # 5'11", 5ft11in, 5'11
    if m:
        ft = int(m.group(1)); inch = int(m.group(3) or 0)
        return (ft*12 + inch) * 0.0254
    m = re.match(r'^(\d+)[\-\.](\d+)$', hs)  # 5-11
    if m:
        ft = int(m.group(1)); inch = int(m.group(2))
        return (ft*12 + inch) * 0.0254
    raise ValueError(f"Cannot parse height: {h}")

def parse_weight(val: str, unit: Optional[str]) -> float:
    w = float(val)
    if unit is None or unit.strip()=="" or unit.lower().startswith("kg"):
        return w
    if unit.lower() in ("lb","lbs","pound","pounds"):
        return w * 0.45359237
    raise ValueError(f"Unknown unit: {unit}")

def kg_to_lb(x: float) -> float:
    return x / 0.45359237

# ----------------- data ingest -----------------
def read_csv_required(path: str) -> tuple[float, float, list[tuple[datetime,float]]]:
    """Return (start_kg, goal_kg, rows[ (date, kg) ... ]) or raise."""
    start_kg = None
    goal_kg = None
    rows: list[tuple[datetime,float]] = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for raw in r:
            if not raw: 
                continue
            key = raw[0].strip().lower()
            if key.startswith("#"):  # comments
                continue
            # detect header "date,weight,unit"
            if key == "date": 
                continue
            if key in ("start_weight", "goal_weight"):
                if len(raw) < 2:
                    raise ValueError(f"{key} row must be: {key},<value>,<unit>")
                val = raw[1].strip()
                unit = raw[2].strip() if len(raw)>=3 else "kg"
                if key == "start_weight": start_kg = parse_weight(val, unit)
                else: goal_kg = parse_weight(val, unit)
                continue
            # data rows
            if len(raw) < 2:
                continue
            try:
                d = datetime.fromisoformat(raw[0].strip())
            except Exception as e:
                raise ValueError(f"Bad date '{raw[0]}' (use YYYY-MM-DD): {e}")
            wkg = parse_weight(raw[1].strip(), raw[2].strip() if len(raw)>=3 else None)
            rows.append((d, wkg))
    if start_kg is None or goal_kg is None:
        raise ValueError("CSV must include start_weight and goal_weight at the top.")
    rows.sort(key=lambda x: x[0])
    if goal_kg <= start_kg:
        raise ValueError(f"Goal weight ({goal_kg:.1f} kg) must be greater than start weight ({start_kg:.1f} kg).")
    if len(rows) < 2:
        raise ValueError("Need at least 2 weigh-ins after the required rows.")
    return start_kg, goal_kg, rows

# ----------------- metrics -----------------
def bmi(weight_kg: float, height_m: float) -> float:
    return weight_kg / (height_m**2)

def bmi_category(b: float) -> str:
    if b < 18.5: return "Underweight"
    if b < 25: return "Normal"
    if b < 30: return "Overweight"
    if b < 35: return "Obesity I"
    if b < 40: return "Obesity II"
    return "Obesity III"

def trend_per_week(dates: List[datetime], kgs: List[float]) -> float:
    if len(kgs) < 2: return 0.0
    x = [(d - dates[0]).days for d in dates]
    y = kgs
    n = len(y)
    meanx = sum(x)/n; meany = sum(y)/n
    num = sum((xi-meanx)*(yi-meany) for xi,yi in zip(x,y))
    den = sum((xi-meanx)**2 for xi in x) or 1e-9
    slope_kg_per_day = num/den
    return slope_kg_per_day * 7.0

# ----------------- visuals -----------------
def svg_segmented_chart_with_goal(dates: List[datetime], kgs: List[float], goal_kg: float, width=800, height=260, margin=30) -> str:
    if len(kgs) < 2:
        return f"<svg width='{width}' height='{height}'><text x='10' y='20' font-family='sans-serif' font-size='14'>Add more data for a chart.</text></svg>"
    min_w, max_w = min(min(kgs), goal_kg), max(max(kgs), goal_kg)
    min_d, max_d = dates[0], dates[-1]
    pad = (max_w - min_w) * 0.1 if max_w>min_w else 1.0
    min_w -= pad; max_w += pad
    def sx(dt: datetime) -> float:
        total = max((max_d - min_d).days, 1)
        return margin + ((dt - min_d).days / total) * (width - 2*margin)
    def sy(w: float) -> float:
        rng = max((max_w - min_w), 1e-9)
        return height - margin - ((w - min_w) / rng) * (height - 2*margin)
    # grid
    y_ticks = 4
    grid = []
    for i in range(y_ticks+1):
        yy = margin + i*(height-2*margin)/y_ticks
        val = max_w - i*(max_w-min_w)/y_ticks
        grid.append(f"<line x1='{margin}' y1='{yy:.1f}' x2='{width-margin}' y2='{yy:.1f}' stroke='#2a2a2f' stroke-width='0.8'/>"
                    f"<text x='8' y='{yy+4:.1f}' font-size='10' font-family='sans-serif' fill='#b6b6b8'>{val:.1f} kg</text>")
    # segments
    segs = []
    for i in range(1, len(kgs)):
        x1, y1 = sx(dates[i-1]), sy(kgs[i-1])
        x2, y2 = sx(dates[i]),   sy(kgs[i])
        color = "var(--gain)" if kgs[i] >= kgs[i-1] else "var(--loss)"
        segs.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='3' stroke-linecap='round'/>")
    # goal line
    gy = sy(goal_kg)
    goal_line = f"<line x1='{margin}' y1='{gy:.1f}' x2='{width-margin}' y2='{gy:.1f}' stroke='#888' stroke-dasharray='4 4'/>" \
                f"<text x='{width-margin}' y='{gy-6:.1f}' font-size='11' font-family='sans-serif' fill='#b6b6b8' text-anchor='end'>Goal {goal_kg:.1f} kg ({kg_to_lb(goal_kg):.1f} lb)</text>"
    # dates
    mid = min_d + (max_d - min_d)/2
    date_labels = [(min_d, sx(min_d)), (mid, sx(mid)), (max_d, sx(max_d))]
    date_text = "".join(f"<text x='{x:.1f}' y='{height-8}' font-size='10' font-family='sans-serif' fill='#b6b6b8' text-anchor='middle'>{dt.date()}</text>"
                        for dt,x in date_labels)
    legend = ("<g transform='translate(0,0)'>"
              "<rect x='0' y='0' width='10' height='10' fill='var(--gain)'/><text x='16' y='9' font-size='11' fill='#b6b6b8'>gain</text>"
              "<rect x='60' y='0' width='10' height='10' fill='var(--loss)'/><text x='76' y='9' font-size='11' fill='#b6b6b8'>loss</text>"
              "</g>")
    return f"""
    <svg width="{width}" height="{height}" role="img" aria-label="Weight over time">
      <rect x="0" y="0" width="{width}" height="{height}" fill="var(--card)" />
      {''.join(grid)}
      {''.join(segs)}
      {goal_line}
      {date_text}
      <g transform="translate({width-160},{20})">{legend}</g>
    </svg>
    """

def progress_bar(percent: float, width: int = 20) -> str:
    pct = max(0.0, percent)
    filled = int(round(min(pct, 100.0) / 100 * width))
    return "[" + "█"*filled + "░"*(width-filled) + f"] {pct:.1f}%"

# ----------------- reports -----------------
def html_report(name: str, height_m: float, start_kg: float, goal_kg: float, rows: List[Tuple[datetime,float]]) -> str:
    dates = [d for d,_ in rows]
    kgs = [w for _,w in rows]
    end_w = kgs[-1]
    total_needed_kg = goal_kg - start_kg
    current_gain_kg = end_w - start_kg
    progress_pct = (current_gain_kg / total_needed_kg) * 100.0
    pace_reg_kg = trend_per_week(dates, kgs)
    pace_kg = (end_w - kgs[0]) / max((dates[-1]-dates[0]).days/7.0, 1e-9)
    end_bmi = bmi(end_w, height_m)

    svg = svg_segmented_chart_with_goal(dates, kgs, goal_kg)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Weight Gain Report — {name}</title>
<style>
  :root {{ --bg:#0b0b0c; --fg:#f2f2f3; --muted:#b6b6b8; --card:#141416; --accent:#ffffff; --gain:#27c93f; --loss:#ff5f56; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  .wrap {{ max-width: 980px; margin: 32px auto; padding: 0 16px; }}
  h1 {{ font-weight: 800; letter-spacing: -0.02em; }}
  .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit,minmax(230px,1fr)); margin: 16px 0 24px; }}
  .card {{ background: var(--card); border: 1px solid #222; border-radius: 16px; padding: 16px; }}
  .kpi {{ font-size: 24px; font-weight: 800; }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  .chart {{ background: var(--card); border: 1px solid #222; border-radius: 16px; padding: 8px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px; border-bottom: 1px solid #222; font-variant-numeric: tabular-nums; }}
  th {{ text-align: left; color: var(--muted); }}
  .pill {{ display:inline-block; padding: 3px 8px; border-radius: 999px; background:#1e1e22; border:1px solid #2a2a2f; font-size:12px; }}
  .bar {{ height: 12px; background:#1e1e22; border:1px solid #2a2a2f; border-radius: 999px; overflow:hidden; }}
  .bar > div {{ height:100%; background: linear-gradient(90deg, var(--gain), #64e384); width:{min(max(progress_pct,0),100):.2f}%; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Weight Gain Report — {name}</h1>
    <div class="grid">
      <div class="card"><div class="muted">Start / Goal / Current</div>
        <div class="kpi">{start_kg:.1f} kg ({kg_to_lb(start_kg):.1f} lb) → {goal_kg:.1f} kg ({kg_to_lb(goal_kg):.1f} lb) → <strong>{end_w:.1f} kg ({kg_to_lb(end_w):.1f} lb)</strong></div></div>
      <div class="card"><div class="muted">Progress to goal</div>
        <div class="kpi">{current_gain_kg:+.1f} / {total_needed_kg:.1f} kg ({kg_to_lb(current_gain_kg):+.1f} / {kg_to_lb(total_needed_kg):.1f} lb) — {progress_pct:.1f}%</div>
        <div class="bar" title="Progress to goal"><div></div></div>
      </div>
      <div class="card"><div class="muted">Pace (overall / trend)</div>
        <div class="kpi">{pace_kg:.2f} / {pace_reg_kg:.2f} kg/week ({kg_to_lb(pace_kg):.2f} / {kg_to_lb(pace_reg_kg):.2f} lb/week)</div></div>
      <div class="card"><div class="muted">BMI</div>
        <div class="kpi">{end_bmi:.1f} <span class="pill">{bmi_category(end_bmi)}</span></div></div>
    </div>

    <div class="chart" aria-label="Weight chart">
      {svg}
    </div>

    <h2>History</h2>
    <table>
      <thead><tr><th>Date</th><th>Weight</th><th>BMI</th><th>Δ from prev</th></tr></thead>
      <tbody>
        {''.join(f"<tr><td>{d.date()}</td><td>{w:.1f} kg ({kg_to_lb(w):.1f} lb)</td><td>{bmi(w, height_m):.1f}</td><td style='color:{'#27c93f' if (i>0 and (w-rows[i-1][1])>=0) else '#ff5f56' if i>0 else 'var(--muted)'}'>{(w-rows[i-1][1]):+0.1f} kg</td></tr>" for i,(d,w) in enumerate(rows))}
      </tbody>
    </table>

    <p class="muted">Green = gain, Red = loss. Dashed line marks your goal weight. Generated by gainer.py — fully offline.</p>
  </div>
</body>
</html>"""
    return html

def print_terminal(name: str, height_m: float, start_kg: float, goal_kg: float, rows: List[Tuple[datetime,float]], min_points: int) -> None:
    if len(rows) < max(2, min_points):
        print(f"Need at least {max(2,min_points)} data points. You have {len(rows)}.")
        return
    dates = [d for d,_ in rows]
    kgs = [w for _,w in rows]
    start_w, end_w = start_kg, kgs[-1]
    total_needed = goal_kg - start_kg
    current_gain = end_w - start_kg
    progress_pct = (current_gain / total_needed) * 100.0
    weeks = max((dates[-1]-dates[0]).days/7.0, 1e-9)
    pace = (end_w - kgs[0]) / weeks
    pace_reg = trend_per_week(dates, kgs)
    start_bmi = bmi(start_w, height_m); end_bmi = bmi(end_w, height_m)
    arrows = "".join("▲" if i>0 and kgs[i]>=kgs[i-1] else "▼" if i>0 else "•" for i in range(len(kgs)))

    print(f"\n=== {name} — Weight Gain Tracker ===")
    print(f"Start / Goal / Now: {start_w:.1f} kg ({kg_to_lb(start_w):.1f} lb) → {goal_kg:.1f} kg ({kg_to_lb(goal_kg):.1f} lb) → {end_w:.1f} kg ({kg_to_lb(end_w):.1f} lb)")
    print(f"Progress to goal  : {current_gain:+.1f}/{total_needed:.1f} kg  ({kg_to_lb(current_gain):+.1f}/{kg_to_lb(total_needed):.1f} lb)  {progress_bar(progress_pct)}")
    print(f"BMI               : {end_bmi:.1f} ({bmi_category(end_bmi)})  [was {start_bmi:.1f}]")
    print(f"Pace              : {pace:.2f} kg/w (overall), {pace_reg:.2f} kg/w (trend) — {kg_to_lb(pace):.2f}/{kg_to_lb(pace_reg):.2f} lb/w")
    print(f"Span              : {(dates[-1]-dates[0]).days} days, {len(kgs)} entries")
    print(f"Trend             : {arrows}\n")

# ----------------- main -----------------
def main(argv: List[str]) -> None:
    args = {"--csv": None, "--height": None, "--name":"You", "--out": None, "--min-points":"3"}
    i=0
    while i < len(argv):
        if argv[i] in args:
            if argv[i] == "--out":
                args["--out"] = argv[i+1] if i+1<len(argv) else None
                i+=2; continue
            args[argv[i]] = argv[i+1] if i+1<len(argv) else None
            i+=2
        else:
            i+=1
    if not args["--csv"] or not args["--height"]:
        print("Usage: python3 gainer.py --csv data.csv --height 180cm [--name You] [--out report.html] [--min-points 3]")
        sys.exit(1)
    try:
        height_m = parse_height(args["--height"])
    except Exception as e:
        print(f"Error: {e}"); sys.exit(1)
    try:
        start_kg, goal_kg, rows = read_csv_required(args["--csv"])
    except Exception as e:
        print(f"CSV error: {e}"); sys.exit(1)
    name = args["--name"] or "You"
    min_points = int(args["--min-points"] or 3)
    if len(rows) < max(2, min_points):
        print(f"Need at least {max(2,min_points)} data points in CSV (have {len(rows)})."); sys.exit(1)
    print_terminal(name, height_m, start_kg, goal_kg, rows, min_points)
    if args["--out"]:
        html = html_report(name, height_m, start_kg, goal_kg, rows)
        with open(args["--out"], "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report written to: {args['--out']}")

if __name__ == "__main__":
    main(sys.argv[1:])
