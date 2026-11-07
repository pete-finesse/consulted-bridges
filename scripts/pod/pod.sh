#!/usr/bin/env bash
# Drive the RunPod pod FROM THE MAC. Set the SSH command RunPod shows, once per shell:
#   export POD_SSH="ssh root@1.2.3.4 -p 22033 -i ~/.ssh/id_ed25519"
# then:
#   ./pod.sh push            copy the code + TwoHopFact to /workspace, run setup_pod.sh
#   ./pod.sh push --code     copy the code only (after editing it here), no setup
#   ./pod.sh run NAME CMD..  run `uv run cb --config $POD_CONFIG CMD..` (default qwen35_4b) on the pod in the background,
#                            log to /workspace/logs/NAME.log, results under results/
#   ./pod.sh status [NAME]   GPU use, running jobs, tail of a log
#   ./pod.sh pull            copy pod results to data/pod-results/ (git-ignored)
#   ./pod.sh ssh [CMD]       a shell (or one command) on the pod, environment loaded
# Uses tar over ssh, so the pod needs no rsync.
set -euo pipefail
: "${POD_SSH:?set POD_SSH to the ssh command RunPod shows}"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)            # repository root
REMOTE=/workspace/consulted-bridges
read -r -a SSH <<< "$POD_SSH"
SSH=("${SSH[@]/#\~/$HOME}" -o ServerAliveInterval=30)
on_pod() { "${SSH[@]}" "$@"; }

case "${1:-}" in
  push)
    echo "copying code and data to the pod..."
    COPYFILE_DISABLE=1 tar -C "$ROOT" --no-mac-metadata --no-xattrs -czf - --exclude='.venv' --exclude='.tiny_cache' --exclude='results' --exclude='*.egg-info' \
        --exclude='__pycache__' --exclude='paper' --exclude='*.pdf' \
        pyproject.toml uv.lock src scripts configs tests data/twohopfact \
      | on_pod "mkdir -p $REMOTE && tar -C $REMOTE --no-same-owner -xzf - && find $REMOTE -name '._*' -delete"
    [[ "${2:-}" == "--code" ]] && { echo "code updated"; exit 0; }
    # detached, so a dropped connection doesn't kill it; then follow its log (Ctrl-C stops following only)
    on_pod "mkdir -p /workspace/logs; cd $REMOTE; nohup bash scripts/pod/setup_pod.sh > /workspace/logs/setup.log 2>&1 < /dev/null &"
    sleep 2
    on_pod "tail -n +1 --pid=\$(pgrep -of setup_pod.sh || echo 999999) -f /workspace/logs/setup.log" || true ;;
  run)
    name=${2:?job name}; shift 2
    cmd="source /workspace/env.sh && nohup uv run cb --config ${POD_CONFIG:-qwen35_4b} $(printf '%q ' "$@") > /workspace/logs/$name.log 2>&1 < /dev/null &"
    on_pod "mkdir -p /workspace/logs && bash -c $(printf '%q' "$cmd")"
    echo "started '$name'; ./pod.sh status $name" ;;
  status)
    on_pod "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader;
            echo jobs:; ps -eo pid,etime,args | grep '[c]b --config' | grep -v 'bash -c' || echo '  none: pod idle';
            ${2:+tail -n 15 /workspace/logs/$2.log}" ;;
  pull)
    mkdir -p "$ROOT/data/pod-results"
    on_pod "tar -C $REMOTE -czf - results" | tar -C "$ROOT/data/pod-results" -xzf -
    on_pod "tar -C /workspace -czf - logs" | tar -C "$ROOT/data/pod-results" -xzf -
    echo "copied to $ROOT/data/pod-results/" ;;
  ssh)
    shift
    if [[ $# -gt 0 ]]; then on_pod "source /workspace/env.sh && $*"; else "${SSH[@]}" -t "source /workspace/env.sh; exec bash -l"; fi ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
