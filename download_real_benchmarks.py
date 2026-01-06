#!/usr/bin/env python3
"""
Download Real Benchmark Data for WebArena and InterCode

This script downloads the official benchmark datasets:
- InterCode: Bash command generation from Princeton NLP
- WebArena: Web navigation tasks

Usage:
    python download_real_benchmarks.py --data-path ./data
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
import urllib.request
import tempfile
import shutil


def download_intercode(data_path: str):
    """Download InterCode bash benchmark from official GitHub."""
    print("\n" + "="*60)
    print("Downloading InterCode Benchmark")
    print("="*60)
    
    intercode_dir = Path(data_path) / "intercode"
    intercode_dir.mkdir(parents=True, exist_ok=True)
    
    # Official InterCode GitHub raw URLs
    base_url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ic_bash"
    
    files_to_download = [
        ("ic_bash.json", "intercode_tasks.json"),
    ]
    
    # Try direct download first
    try:
        print("Attempting direct download from GitHub...")
        url = f"{base_url}/ic_bash.json"
        output_file = intercode_dir / "intercode_tasks.json"
        
        urllib.request.urlretrieve(url, output_file)
        
        # Verify and convert format
        with open(output_file) as f:
            data = json.load(f)
        
        # Convert to our format
        tasks = []
        for i, item in enumerate(data):
            query = item.get("query", item.get("instruction", item.get("input", "")))
            expected = item.get("gold", item.get("output", item.get("command", "")))
            
            if query and expected:
                tasks.append({
                    "id": f"intercode_{i}",
                    "query": query,
                    "expected_command": expected
                })
        
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"✓ Downloaded {len(tasks)} InterCode tasks")
        return True
        
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    # Try cloning the repo
    try:
        print("Attempting git clone...")
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["git", "clone", "--depth", "1", 
                 "https://github.com/princeton-nlp/intercode.git", tmpdir],
                check=True, capture_output=True
            )
            
            # Find and copy the data
            src_file = Path(tmpdir) / "data" / "ic_bash" / "ic_bash.json"
            if src_file.exists():
                with open(src_file) as f:
                    data = json.load(f)
                
                tasks = []
                for i, item in enumerate(data):
                    query = item.get("query", item.get("instruction", ""))
                    expected = item.get("gold", item.get("command", ""))
                    if query and expected:
                        tasks.append({
                            "id": f"intercode_{i}",
                            "query": query,
                            "expected_command": expected
                        })
                
                output_file = intercode_dir / "intercode_tasks.json"
                with open(output_file, 'w') as f:
                    json.dump(tasks, f, indent=2)
                
                print(f"✓ Downloaded {len(tasks)} InterCode tasks via git")
                return True
                
    except Exception as e:
        print(f"Git clone failed: {e}")
    
    # Try NL2Bash as fallback (real data)
    try:
        print("Trying NL2Bash dataset from HuggingFace...")
        from datasets import load_dataset
        
        dataset = load_dataset("neulab/nl2bash", split="test", trust_remote_code=True)
        
        tasks = []
        for i, sample in enumerate(dataset):
            query = sample.get("invocation", sample.get("nl", ""))
            cmd = sample.get("cmd", sample.get("bash", ""))
            if query and cmd:
                tasks.append({
                    "id": f"intercode_{i}",
                    "query": query,
                    "expected_command": cmd
                })
        
        if tasks:
            output_file = intercode_dir / "intercode_tasks.json"
            with open(output_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            print(f"✓ Downloaded {len(tasks)} tasks from NL2Bash")
            return True
            
    except Exception as e:
        print(f"NL2Bash failed: {e}")
    
    print("✗ Could not download InterCode data")
    return False


def download_webarena(data_path: str):
    """Download WebArena benchmark from official GitHub."""
    print("\n" + "="*60)
    print("Downloading WebArena Benchmark")
    print("="*60)
    
    webarena_dir = Path(data_path) / "webarena"
    webarena_dir.mkdir(parents=True, exist_ok=True)
    
    # Official WebArena GitHub raw URLs for task configs
    base_url = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files"
    
    task_files = [
        "test_shopping.json",
        "test_reddit.json",
        "test_gitlab.json",
        "test_wikipedia.json",
        "test_map.json",
    ]
    
    all_tasks = []
    task_id = 0
    
    for task_file in task_files:
        try:
            url = f"{base_url}/{task_file}"
            print(f"Downloading {task_file}...")
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                items = list(data.values()) if not isinstance(list(data.values())[0], dict) else [data]
            else:
                items = data
            
            for item in items:
                if isinstance(item, dict):
                    intent = item.get("intent", item.get("task", item.get("goal", "")))
                    if intent:
                        all_tasks.append({
                            "id": f"webarena_{task_id}",
                            "query": intent,
                            "site": task_file.replace("test_", "").replace(".json", ""),
                            "expected_actions": item.get("action_sequence", []),
                            "eval_type": item.get("eval", {}).get("eval_types", [])
                        })
                        task_id += 1
                        
        except Exception as e:
            print(f"  Failed to download {task_file}: {e}")
            continue
    
    if all_tasks:
        output_file = webarena_dir / "webarena_tasks.json"
        with open(output_file, 'w') as f:
            json.dump(all_tasks, f, indent=2)
        print(f"✓ Downloaded {len(all_tasks)} WebArena tasks")
        return True
    
    # Try Mind2Web as fallback (real web navigation data)
    try:
        print("Trying Mind2Web dataset from HuggingFace...")
        from datasets import load_dataset
        
        dataset = load_dataset("osunlp/Mind2Web", split="test", trust_remote_code=True)
        
        tasks = []
        for i, sample in enumerate(dataset):
            if i >= 500:  # Limit to 500 tasks
                break
            task = sample.get("confirmed_task", sample.get("task", ""))
            if task:
                tasks.append({
                    "id": f"webarena_{i}",
                    "query": task,
                    "site": sample.get("website", sample.get("domain", "")),
                    "expected_actions": [],
                    "annotation_id": sample.get("annotation_id", "")
                })
        
        if tasks:
            output_file = webarena_dir / "webarena_tasks.json"
            with open(output_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            print(f"✓ Downloaded {len(tasks)} tasks from Mind2Web")
            return True
            
    except Exception as e:
        print(f"Mind2Web failed: {e}")
    
    print("✗ Could not download WebArena data")
    return False


def download_spider(data_path: str):
    """Download Spider SQL benchmark."""
    print("\n" + "="*60)
    print("Downloading Spider SQL Benchmark")
    print("="*60)
    
    spider_dir = Path(data_path) / "spider2"
    spider_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Try official Spider
        dataset = load_dataset("spider", split="validation", trust_remote_code=True)
        
        tasks = []
        for i, sample in enumerate(dataset):
            tasks.append({
                "id": f"spider_{i}",
                "query": sample.get("question", ""),
                "expected_sql": sample.get("query", ""),
                "database": sample.get("db_id", "")
            })
        
        output_file = spider_dir / "spider2_tasks.json"
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"✓ Downloaded {len(tasks)} Spider tasks")
        return True
        
    except Exception as e:
        print(f"Spider download failed: {e}")
        return False


def download_gorilla(data_path: str):
    """Download Gorilla API benchmark."""
    print("\n" + "="*60)
    print("Downloading Gorilla API Benchmark")
    print("="*60)
    
    gorilla_dir = Path(data_path) / "gorilla"
    gorilla_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Try Gorilla OpenFunctions
        dataset = load_dataset("gorilla-llm/gorilla-openfunctions-v2", split="train", trust_remote_code=True)
        
        tasks = []
        for i, sample in enumerate(dataset):
            if i >= 500:  # Limit
                break
            tasks.append({
                "id": f"gorilla_{i}",
                "query": sample.get("question", sample.get("instruction", "")),
                "expected": sample.get("answer", sample.get("output", "")),
                "functions": sample.get("functions", [])
            })
        
        output_file = gorilla_dir / "gorilla_tasks.json"
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"✓ Downloaded {len(tasks)} Gorilla tasks")
        return True
        
    except Exception as e:
        print(f"Gorilla download failed: {e}")
        return False


def verify_downloads(data_path: str):
    """Verify all downloads and show summary."""
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    benchmarks = {
        "gorilla": "gorilla/gorilla_tasks.json",
        "spider2": "spider2/spider2_tasks.json",
        "webarena": "webarena/webarena_tasks.json",
        "intercode": "intercode/intercode_tasks.json",
    }
    
    for name, path in benchmarks.items():
        full_path = Path(data_path) / path
        if full_path.exists():
            with open(full_path) as f:
                tasks = json.load(f)
            print(f"✓ {name}: {len(tasks)} tasks")
        else:
            print(f"✗ {name}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Download real benchmark datasets")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to store benchmark data")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["gorilla", "spider", "webarena", "intercode"],
                       help="Benchmarks to download")
    args = parser.parse_args()
    
    print("="*60)
    print("DOWNLOADING REAL BENCHMARK DATA")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Benchmarks: {args.benchmarks}")
    
    os.makedirs(args.data_path, exist_ok=True)
    
    results = {}
    
    if "gorilla" in args.benchmarks:
        results["gorilla"] = download_gorilla(args.data_path)
    
    if "spider" in args.benchmarks:
        results["spider"] = download_spider(args.data_path)
    
    if "webarena" in args.benchmarks:
        results["webarena"] = download_webarena(args.data_path)
    
    if "intercode" in args.benchmarks:
        results["intercode"] = download_intercode(args.data_path)
    
    verify_downloads(args.data_path)
    
    # Return success if at least some downloaded
    success_count = sum(results.values())
    print(f"\n✓ Successfully downloaded {success_count}/{len(results)} benchmarks")
    
    return success_count > 0


if __name__ == "__main__":
    main()
