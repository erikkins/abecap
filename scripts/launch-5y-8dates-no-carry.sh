#!/bin/bash
# Ablation control for the CB pause-carries-periods behavior. Same 8 start
# dates × 5y windows as launch-5y-8dates.sh, but with --cb-pause-no-carry-
# periods set so a CB pause clears at the period boundary (mimicking the
# pre-Apr 28 implicit behavior).
#
# Each run writes pickle + json directly to its per-date dir via
# WF_RESULT_PICKLE / WF_RESULT_JSON env vars, so this script is safe to
# run concurrently with launch-5y-8dates.sh.
#
# Output dir convention: /tmp/wf_5y_<start>_no_carry/
# Aggregate CSV: /tmp/wf_5y_8dates_no_carry_summary.csv
#
# Idempotent — re-running skips dates that already have wf_run_result.json.
# Per-run failures don't abort the whole script.

cd /Users/erikkins/CODE/stocker-app
source backend/venv/bin/activate

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

SUMMARY_CSV="/tmp/wf_5y_8dates_no_carry_summary.csv"
echo "start_date,end_date,total_return_pct,sharpe_ratio,max_drawdown_pct,benchmark_return_pct,total_trades" > "$SUMMARY_CSV"

T_START=$(date +%s)
echo "Launching 8 sequential 5y NO-CARRY ablation runs at $(date)"
echo "Per-job logs at /tmp/wf_5y_<start>_no_carry/run.log"
echo

i=0
for pair in "${DATES[@]}"; do
  i=$((i+1))
  START="${pair%%:*}"
  END="${pair##*:}"
  RUNDIR="/tmp/wf_5y_${START}_no_carry"
  mkdir -p "$RUNDIR"

  if [ -f "$RUNDIR/wf_run_result.json" ]; then
    echo "[$(date '+%H:%M:%S')] Run $i/8 (no-carry): $START → $END  (already complete, skipping)"
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

  echo "[$(date '+%H:%M:%S')] Run $i/8 (no-carry): $START → $END"

  WF_RESULT_PICKLE="$RUNDIR/wf_run_result.pkl" \
  WF_RESULT_JSON="$RUNDIR/wf_run_result.json" \
  caffeinate -i python3 scripts/local_wf_runner.py \
    --pickle backend/data/all_data_11y.pkl.gz \
    --start "$START" \
    --end "$END" \
    --strategy-id 5 \
    --max-symbols 200 \
    --no-ai \
    --cb-pause-no-carry-periods \
    > "$RUNDIR/run.log" 2>&1 || echo "  (runner exited non-zero — checking for partial result)"

  if [ -f "$RUNDIR/wf_run_result.json" ]; then
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
echo "All 8 no-carry runs complete in $(( (T_END - T_START) / 60 )) min."
echo "No-carry summary CSV: $SUMMARY_CSV"
echo
column -t -s, "$SUMMARY_CSV"
echo
echo "Aggregate (no-carry):"
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

echo
echo "=================================================================="
echo "TO COMPARE WITH-CARRY vs NO-CARRY (run after both finish):"
echo "=================================================================="
echo "  python3 -c \"
import csv
def avg(field, path):
    rows = list(csv.DictReader(open(path)))
    vals = [float(r[field]) for r in rows if r[field] not in ('', 'ERROR')]
    return sum(vals)/len(vals) if vals else float('nan')

for f in ['total_return_pct','sharpe_ratio','max_drawdown_pct']:
    a = avg(f, '/tmp/wf_5y_8dates_summary.csv')
    b = avg(f, '/tmp/wf_5y_8dates_no_carry_summary.csv')
    print(f'  {f:24s} carry={a:8.2f} | no-carry={b:8.2f} | delta={a-b:+.2f}')
\""
