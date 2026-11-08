#!/usr/bin/env bash
# tools/lab_unit.sh - tools/lab.py on a PC whose Maya server runs as a systemd user unit, which holds the GPU memory a
# run needs. The run is a transient unit of its own (maya-lab): it stops the server's unit first (ExecStartPre) and
# starts it again when the run ends - finished, failed, killed, or past --max-sec (RuntimeMaxSec) - whatever happens to
# this shell or its ssh session (ExecStopPost). A server that is answering a request, or had one in the last
# MAYA_LAB_QUIET seconds (180: an agent between its turns), is left alone unless --force; --wait N tries again every
# 30 s for up to N seconds. --follow prints the run's lines until it ends (else: journalctl, below).
#
#   tools/lab_unit.sh [--unit maya-ds4.service] [--max-sec 900] [--wait 3600] [--follow] [--force] \
#       -- run --config maya-ds4-hip.json --label base
#   journalctl --user -u maya-lab -f -o cat      # follow it; its last line starts with "lab: end"
#
# MAYA_LAB_PY: the Python to run lab.py with (default: .venv/bin/python here, else python3). MAYA_LAB_URL (default
# http://127.0.0.1:8080) and MAYA_LAB_KEY or MAYA_LAB_KEY_FILE (a file with STRATA_API_KEY=...): where to ask the server
# whether it is busy (GET /v1/status: activity.in_flight and last_request_at).
set -euo pipefail

unit=maya-ds4.service
max=900
wait_s=0
follow=0
force=0
while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in
    --unit) unit=$2; shift 2 ;;
    --max-sec) max=$2; shift 2 ;;
    --wait) wait_s=$2; shift 2 ;;
    --follow) follow=1; shift ;;
    --force) force=1; shift ;;
    *) echo "lab_unit.sh: unknown option $1 (then -- and lab.py's arguments)" >&2; exit 2 ;;
  esac
done
[[ ${1-} == -- ]] || { echo "lab_unit.sh: -- and lab.py's arguments are missing" >&2; exit 2; }
shift

here=$(cd "$(dirname "$0")/.." && pwd)
py=${MAYA_LAB_PY:-}
[[ -n $py ]] || { [[ -x $here/.venv/bin/python ]] && py=$here/.venv/bin/python || py=$(command -v python3); }
quiet=${MAYA_LAB_QUIET:-180}

# 0: the server may be stopped now (not running, or idle and quiet); 1: not now (why: on stderr)
guard() {
  systemctl --user is-active --quiet "$unit" || return 0
  [[ $force == 1 ]] && return 0
  local key=${MAYA_LAB_KEY:-} kf m busy since
  if [[ -z $key ]]; then
    kf=${MAYA_LAB_KEY_FILE:-$HOME/.config/maya/api-key.env}
    [[ -r $kf ]] && key=$(sed -n 's/^STRATA_API_KEY=//p' "$kf" | head -1)
  fi
  m=$(curl -fsS -m 5 ${key:+-H "Authorization: Bearer $key"} "${MAYA_LAB_URL:-http://127.0.0.1:8080}/v1/status" || true)
  # in flight now, and seconds since the last request began: an agent's next turn comes a few seconds after its last
  read -r busy since < <(printf '%s' "$m" | "$py" -c 'import json,sys,time
try:
    a = json.load(sys.stdin)["activity"]; t = a.get("last_request_at")
    print(int(a.get("in_flight") or 0), int(time.time() - t) if t else 1 << 30)
except Exception: print(-1, 0)')
  if [[ $busy == -1 ]]; then
    echo "lab_unit.sh: $unit is running but its /v1/status did not answer - still loading?" >&2
  elif [[ $busy -gt 0 ]]; then
    echo "lab_unit.sh: $unit is answering $busy request(s) now" >&2
  elif [[ $since -lt $quiet ]]; then
    echo "lab_unit.sh: $unit had a request ${since} s ago - it runs after ${quiet} s of quiet (MAYA_LAB_QUIET)" >&2
  else
    return 0
  fi
  return 1
}

if systemctl --user is-active --quiet maya-lab.service; then
  echo "lab_unit.sh: a run is going on already - journalctl --user -u maya-lab -f -o cat" >&2
  exit 1
fi
until_t=$(( $(date +%s) + wait_s ))
until guard; do
  if [[ $(date +%s) -ge $until_t ]]; then
    echo "lab_unit.sh: not started$([[ $wait_s -gt 0 ]] && echo " in ${wait_s} s") (--wait N tries again for N s," \
         "--force stops the server anyway)" >&2
    exit 1
  fi
  sleep 30
done
systemctl --user reset-failed maya-lab.service 2>/dev/null || true

was_active=0
systemctl --user is-active --quiet "$unit" && was_active=1
props=(-p "RuntimeMaxSec=$max" -p TimeoutStopSec=90 -p KillMode=control-group)
if [[ $was_active == 1 ]]; then   # (a server that was not running stays stopped)
  props+=(-p "ExecStartPre=/usr/bin/systemctl --user stop $unit"
          -p "ExecStopPost=/usr/bin/systemctl --user --no-block start $unit")
fi
since_t=$(date '+%Y-%m-%d %H:%M:%S')
systemd-run --user --unit=maya-lab --collect --quiet "${props[@]}" --working-directory="$here" \
  -- "$py" -u tools/lab.py "$@"
echo "lab_unit.sh: started (the server $([[ $was_active == 1 ]] && echo "comes back after it" || echo "was not running"))"
if [[ $follow == 0 ]]; then
  echo "lab_unit.sh: journalctl --user -u maya-lab -f -o cat"
  exit 0
fi
journalctl --user -u maya-lab -f -o cat --since "$since_t" &
jp=$!
while systemctl --user is-active --quiet maya-lab.service; do sleep 2; done
sleep 1
kill "$jp" 2>/dev/null || true
echo "lab_unit.sh: the run ended$([[ $was_active == 1 ]] && echo "; $unit is starting again")"
