#!/bin/bash
# =============================================================================
# DOWNLOAD REAL BENCHMARK DATA
# Run this script on your Lightning.ai studio to get official benchmark data
# =============================================================================

set -e

DATA_DIR="${1:-./data}"
echo "Downloading real benchmark data to: $DATA_DIR"

# -----------------------------------------------------------------------------
# 1. INTERCODE - Official Princeton NLP Bash Benchmark
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Downloading InterCode (Princeton NLP)"
echo "=========================================="

mkdir -p "$DATA_DIR/intercode"

# Clone InterCode repo and extract bash data
if [ ! -f "$DATA_DIR/intercode/intercode_tasks.json" ]; then
    echo "Cloning InterCode repository..."
    git clone --depth 1 https://github.com/princeton-nlp/intercode.git /tmp/intercode_repo
    
    # The bash benchmark is in data/ic_bash/
    if [ -f "/tmp/intercode_repo/data/ic_bash/ic_bash.json" ]; then
        cp /tmp/intercode_repo/data/ic_bash/ic_bash.json "$DATA_DIR/intercode/ic_bash_raw.json"
        
        # Convert to our format
        python3 << 'EOF'
import json

with open("$DATA_DIR/intercode/ic_bash_raw.json".replace("$DATA_DIR", "${DATA_DIR}")) as f:
    raw_data = json.load(f)

tasks = []
for i, item in enumerate(raw_data):
    task = {
        "id": f"intercode_{i}",
        "query": item.get("query", item.get("instruction", item.get("input", ""))),
        "expected_command": item.get("gold", item.get("output", item.get("command", ""))),
    }
    if task["query"] and task["expected_command"]:
        tasks.append(task)

output_path = "$DATA_DIR/intercode/intercode_tasks.json".replace("$DATA_DIR", "${DATA_DIR}")
with open(output_path, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Converted {len(tasks)} InterCode tasks")
EOF
        
        echo "✓ InterCode downloaded: $(cat $DATA_DIR/intercode/intercode_tasks.json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'  ) tasks"
    else
        echo "InterCode data file not found in repo, trying direct download..."
        curl -L "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ic_bash/ic_bash.json" -o "$DATA_DIR/intercode/ic_bash_raw.json"
    fi
    
    rm -rf /tmp/intercode_repo
else
    echo "InterCode already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 2. WEBARENA - Official Web Navigation Benchmark  
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Downloading WebArena (CMU)"
echo "=========================================="

mkdir -p "$DATA_DIR/webarena"

if [ ! -f "$DATA_DIR/webarena/webarena_tasks.json" ]; then
    echo "Cloning WebArena repository..."
    git clone --depth 1 https://github.com/web-arena-x/webarena.git /tmp/webarena_repo
    
    # WebArena config files contain the tasks
    echo "Extracting WebArena tasks..."
    python3 << 'PYEOF'
import json
import os
from pathlib import Path

webarena_dir = Path("/tmp/webarena_repo/config_files")
output_dir = Path("${DATA_DIR}/webarena")

all_tasks = []
task_id = 0

# Process all test config files
config_files = list(webarena_dir.glob("*.json"))
print(f"Found {len(config_files)} config files")

for config_file in sorted(config_files):
    try:
        with open(config_file) as f:
            data = json.load(f)
        
        # Handle both list and single task formats
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        
        for item in items:
            if isinstance(item, dict) and "intent" in item:
                task = {
                    "id": f"webarena_{task_id}",
                    "query": item.get("intent", ""),
                    "site": item.get("sites", [config_file.stem.replace("test_", "")])[0] if item.get("sites") else config_file.stem.replace("test_", ""),
                    "start_url": item.get("start_url", ""),
                    "eval_types": item.get("eval", {}).get("eval_types", []),
                    "reference_answers": item.get("eval", {}).get("reference_answers", {}),
                }
                all_tasks.append(task)
                task_id += 1
                
    except Exception as e:
        print(f"Error processing {config_file}: {e}")

output_file = output_dir / "webarena_tasks.json"
with open(output_file, "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"Extracted {len(all_tasks)} WebArena tasks")
PYEOF

    rm -rf /tmp/webarena_repo
    echo "✓ WebArena downloaded: $(cat $DATA_DIR/webarena/webarena_tasks.json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))') tasks"
else
    echo "WebArena already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 3. SPIDER - SQL Benchmark (via HuggingFace)
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Downloading Spider SQL Benchmark"
echo "=========================================="

mkdir -p "$DATA_DIR/spider2"

if [ ! -f "$DATA_DIR/spider2/spider2_tasks.json" ]; then
    python3 << 'PYEOF'
import json
from datasets import load_dataset

print("Loading Spider from HuggingFace...")
dataset = load_dataset("spider", split="validation", trust_remote_code=True)

tasks = []
for i, sample in enumerate(dataset):
    tasks.append({
        "id": f"spider_{i}",
        "query": sample.get("question", ""),
        "expected_sql": sample.get("query", ""),
        "database": sample.get("db_id", "")
    })

with open("${DATA_DIR}/spider2/spider2_tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Downloaded {len(tasks)} Spider tasks")
PYEOF
    echo "✓ Spider downloaded"
else
    echo "Spider already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 4. GORILLA - API Benchmark (via HuggingFace)
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Downloading Gorilla API Benchmark"
echo "=========================================="

mkdir -p "$DATA_DIR/gorilla"

if [ ! -f "$DATA_DIR/gorilla/gorilla_tasks.json" ]; then
    python3 << 'PYEOF'
import json
from datasets import load_dataset

print("Loading Gorilla from HuggingFace...")
try:
    dataset = load_dataset("gorilla-llm/gorilla-openfunctions-v2", split="train", trust_remote_code=True)
    
    tasks = []
    for i, sample in enumerate(dataset):
        if i >= 500:  # Limit to 500 tasks
            break
        tasks.append({
            "id": f"gorilla_{i}",
            "query": sample.get("question", sample.get("instruction", "")),
            "expected": sample.get("answer", sample.get("output", "")),
            "functions": sample.get("functions", [])
        })
    
    with open("${DATA_DIR}/gorilla/gorilla_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Downloaded {len(tasks)} Gorilla tasks")
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative Gorilla dataset...")
    
    dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train", trust_remote_code=True)
    tasks = []
    for i, sample in enumerate(dataset):
        if i >= 500:
            break
        tasks.append({
            "id": f"gorilla_{i}",
            "query": sample.get("question", ""),
            "expected": sample.get("answer", ""),
        })
    
    with open("${DATA_DIR}/gorilla/gorilla_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Downloaded {len(tasks)} Gorilla tasks (alternative)")
PYEOF
    echo "✓ Gorilla downloaded"
else
    echo "Gorilla already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
echo ""

for benchmark in gorilla spider2 webarena intercode; do
    file="$DATA_DIR/$benchmark/${benchmark}_tasks.json"
    if [ -f "$file" ] || [ -f "$DATA_DIR/$benchmark/$(ls $DATA_DIR/$benchmark/*.json 2>/dev/null | head -1)" ]; then
        count=$(cat $DATA_DIR/$benchmark/*.json 2>/dev/null | python3 -c 'import json,sys; data=json.load(sys.stdin); print(len(data) if isinstance(data,list) else 1)' 2>/dev/null || echo "?")
        echo "✓ $benchmark: $count tasks"
    else
        echo "✗ $benchmark: NOT FOUND"
    fi
done

echo ""
echo "Data saved to: $DATA_DIR"
