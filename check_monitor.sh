#!/bin/bash
# Quick check of monitor status
echo "=== Monitor Process ==="
ps aux | grep live_model_monitor | grep -v grep || echo "NOT RUNNING"
echo ""
echo "=== Latest Data ==="
if [ -f data/model_monitor.jsonl ]; then
    lines=$(wc -l < data/model_monitor.jsonl)
    echo "Log lines: $lines"
    if [ "$lines" -gt 0 ]; then
        echo "Latest entry:"
        tail -1 data/model_monitor.jsonl | python3 -m json.tool 2>/dev/null | head -20
    fi
else
    echo "No data yet"
fi
echo ""
echo "=== Tmux Last Lines ==="
tmux capture-pane -t test-monitor -p -S -5 2>/dev/null || echo "No tmux session"
