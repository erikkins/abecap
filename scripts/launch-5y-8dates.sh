#!/bin/bash
# 5-year walk-forward across 8 start dates, fixed-params (no AI), on the
# clean 11y pickle. Generates the data for the "5y track record graph" —
# matching the marketing format (§15) of "avg of 8 start dates."
#
# Each run produces:
#   /tmp/wf_5y_<start_date>/wf_run_result.pkl  (full result object)
#   /tmp/wf_5y_<start_date>/wf_run_result.json (scalar summary)
#   /tmp/wf_5y_8dates_summary.csv (aggregated table)
#
# Idempotent — re-running skips dates that already have a wf_run_result.json.
# Per-run failures don't abort the whole script (no `set -e` on the loop).
#
# Each run is ~5 min (no-ai). 8 runs sequential = ~40 min total.

cd /Users/erikkins/CODE/stocker-app
source backend/venv/bin/activate

# 8 start dates spread Jan-Apr 2021. End dates = start + 5y, all land in
# Jan-Apr 2026 (well within pickle data through Apr 27).
DATES=(
  "2021-01-04:2026-01-04"
  "2021-01-18:2026-01-18"
  "2021-01-25:2026-01-25"
  "2021-02-01:2026-02-01"
  "2021-02-08:2026-02-08"
  "2021-02-15:2026-02-15"
  "2021-03-01:2026-03-01"
  "2021-04-01:2026-04-01"
)

SUMMARY_CSV="/tmp/wf_5y_8dates_summary.csv"
echo "start_date,end_date,total_return_pct,sharpe_ratio,max_drawdown_pct,benchmark_return_pct,total_trades" > "$SUMMARY_CSV"

T_START=$(date +%s)
echo "Launching 8 sequential 5y fixed-params WF runs at $(date)"
echo "Progress will be visible at /tmp/wf_5y_<start>/run.log per job."
echo

i=0
for pair in "${DATES[@]}"; do
  i=$((i+1))
  START="${pair%%:*}"
  END="${pair##*:}"
  RUNDIR="/tmp/wf_5y_${START}"
  mkdir -p "$RUNDIR"

  # Idempotent skip: if this date already has a result, harvest its row to
  # the aggregate CSV and move on. Lets us re-fire the launcher without
  # re-running already-completed dates.
  if [ -f "$RUNDIR/wf_run_result.json" ]; then
    echo "[$(date '+%H:%M:%S')] Run $i/8: $START → $END  (already complete, skipping)"
    python3 -c "
import json
try:
    d = json.load(open('$RUNDIR/wf_run_result.json'))
    print(','.join([
        '$START', '$END',
        str(d.get('total_return_pct', '')),
        str(d.get('sharpe_ratio', '')),
        str(d.get('max_drawdown_pct', '')),
        str(d.get('benchmark_return_pct', '')),
        str(d.get('total_trades', '')),
    ]))
except Exception:
    print('$START,$END,ERROR,ERROR,ERROR,ERROR,ERROR')" >> "$SUMMARY_CSV"
    continue
  fi

  echo "[$(date '+%H:%M:%S')] Run $i/8: $START → $END"

  # Run the WF — local_wf_runner pickles result to default paths; we copy
  # them to the per-date dir so they don't overwrite each other.
  # `|| true` so a per-run failure doesn't abort the whole loop.
  caffeinate -i python3 scripts/local_wf_runner.py \
    --pickle backend/data/all_data_11y.pkl.gz \
    --start "$START" \
    --end "$END" \
    --strategy-id 5 \
    --max-symbols 200 \
    --no-ai \
    > "$RUNDIR/run.log" 2>&1 || echo "  (runner exited non-zero — checking for partial result)"

  # Copy result + summary out of /tmp default into per-date dir.
  # The runner pickles BEFORE its post-processing crashes, so partial-failure
  # runs still leave a valid result on disk.
  if [ -f /tmp/wf_run_result.pkl ]; then
    cp /tmp/wf_run_result.pkl "$RUNDIR/wf_run_result.pkl"
  fi
  if [ -f /tmp/wf_run_result.json ]; then
    cp /tmp/wf_run_result.json "$RUNDIR/wf_run_result.json"
    python3 -c "
import json
d = json.load(open('$RUNDIR/wf_run_result.json'))
print(','.join([
    '$START',
    '$END',
    str(d.get('total_return_pct', '')),
    str(d.get('sharpe_ratio', '')),
    str(d.get('max_drawdown_pct', '')),
    str(d.get('benchmark_return_pct', '')),
    str(d.get('total_trades', '')),
]))" >> "$SUMMARY_CSV"
  else
    echo "$START,$END,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY_CSV"
  fi
done

T_END=$(date +%s)
echo
echo "All 8 runs complete in $(( (T_END - T_START) / 60 )) min."
echo "Summary CSV: $SUMMARY_CSV"
echo
column -t -s, "$SUMMARY_CSV"
echo
echo "Aggregate (avg + spread):"
python3 -c "
import csv
rows = list(csv.DictReader(open('$SUMMARY_CSV')))
def to_f(x):
    try: return float(x)
    except: return None
def stats(field):
    vals = [to_f(r[field]) for r in rows if to_f(r[field]) is not None]
    if not vals: return 'no data'
    return f'avg={sum(vals)/len(vals):.2f}, min={min(vals):.2f}, max={max(vals):.2f}, n={len(vals)}'
print(f'  total_return_pct      : {stats(\"total_return_pct\")}')
print(f'  sharpe_ratio          : {stats(\"sharpe_ratio\")}')
print(f'  max_drawdown_pct      : {stats(\"max_drawdown_pct\")}')
print(f'  benchmark_return_pct  : {stats(\"benchmark_return_pct\")}')
print(f'  total_trades          : {stats(\"total_trades\")}')
"
