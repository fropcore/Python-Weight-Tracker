# Weight Gain Tracker (Pro-Gain)

A tiny, dependency-free Python tool that celebrates **gaining**. It reads a CSV, prints terminal stats + motivation, and can generate a **shareable HTML** report with a dark theme, green-for-gain/red-for-loss segments, and a **goal line**.

## CSV Format (required keys at top)

```csv
start_weight,70,kg
goal_weight,85,kg
date,weight,unit
2025-07-01,70,kg
2025-07-08,71,kg
2025-07-15,72,kg
```

- Units can be `kg` or `lb`. Missing `unit` on data rows defaults to `kg`.
- `start_weight` and `goal_weight` are **required**. Goal must be greater than start.

## Usage

```bash
python3 gainer.py --csv sample/sample_weights.csv --height 180cm --name "Bob" --out report.html
# or imperial height:
python3 gainer.py --csv sample/sample_weights.csv --height "5'11"" --name "Bob" --out report.html
```

### What you get

`- **Terminal**: start/goal/current, BMI, pace/week (overall & trend), and a nice progress bar.
- **HTML**: KPIs, progress bar, dual-units (kg/lb), segmented chart (green gain, red loss), and a dashed goal line.

All offline. Share the HTML as-is.

## License

MIT
