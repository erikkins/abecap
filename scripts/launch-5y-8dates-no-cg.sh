#!/bin/bash
# True NO-CG ablation: same 8 start dates × 5y windows as launch-5y-8dates.sh,
# but with --disable-cg set so Cascade Guard / circuit breaker NEVER fires.
# The strategy trades through every panic event without pausing.
#
# Compared against the canonical (with-CG) carry-on results, the delta tells us
# the true return contribution and drawdown impact of Cascade Guard on clean
# data. Replaces the unreliable Apr 19 +297% / +384% / +87pp claim that was
# computed on indicator-corrupted data.
#
# Output dir: /tmp/wf_5y_<start>_no_cg/
# Aggregate CSV: /tmp/wf_5y_8dates_no_cg_summary.csv
#
# Idempotent — re-running skips dates already complete. Per-run failures don't
# abort the whole script.

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

SUMMARY_CSV="/tmp/wf_5y_8dates_no_cg_summary.csv"
echo "start_date,end_date,total_return_pct,sharpe_ratio,max_drawdown_pct,benchmark_return_pct,total_trades" > "$SUMMARY_CSV"

T_START=$(date +%s)
echo "Launching multiple sequential 5y NO-CG ablation runs at $(date)"
echo "Per-job logs at /tmp/wf_5y_<start>_no_cg/run.log"
echo

i=0
for pair in "${DATES[@]}"; do
  i=$((i+1))
  START="${pair%%:*}"
  END="${pair##*:}"
  RUNDIR="/tmp/wf_5y_${START}_no_cg"
  mkdir -p "$RUNDIR"

  if [ -f "$RUNDIR/wf_run_result.json" ]; then
    echo "[$(date '+%H:%M:%S')] Run $i (no-cg): $START → $END  (already complete, skipping)"
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

  echo "[$(date '+%H:%M:%S')] Run $i (no-cg): $START → $END"

  WF_RESULT_PICKLE="$RUNDIR/wf_run_result.pkl" \
  WF_RESULT_JSON="$RUNDIR/wf_run_result.json" \
  caffeinate -i python3 scripts/local_wf_runner.py \
    --pickle backend/data/all_data_11y.pkl.gz \
    --start "$START" \
    --end "$END" \
    --strategy-id 5 \
    --max-symbols 200 \
    --no-ai \
    --disable-cg \
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
echo "All no-CG runs complete in $(( (T_END - T_START) / 60 )) min."
echo "No-CG summary CSV: $SUMMARY_CSV"
echo
column -t -s, "$SUMMARY_CSV"
echo
echo "Aggregate (no-CG):"
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
"

echo
echo "=================================================================="
echo "TO COMPARE WITH-CG (canonical) vs NO-CG:"
echo "=================================================================="
echo "  python3 -c \"
import csv
def avg(field, path):
    rows = list(csv.DictReader(open(path)))
    vals = [float(r[field]) for r in rows if r[field] not in ('', 'ERROR')]
    return sum(vals)/len(vals) if vals else float('nan')

for f in ['total_return_pct','sharpe_ratio','max_drawdown_pct']:
    a = avg(f, '/tmp/wf_5y_8dates_summary.csv')
    b = avg(f, '/tmp/wf_5y_8dates_no_cg_summary.csv')
    print(f'  {f:24s} with-CG={a:8.2f} | no-CG={b:8.2f} | CG-delta={a-b:+.2f}')
\""
