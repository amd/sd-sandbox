# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
"""
Post-process benchmark result files and generate CSV summaries.

Parses pipeline_results_benchmark_*.txt files from the results directory
and extracts key performance metrics into a structured CSV format.

Output CSV columns:
- Model Name: Pipeline/model identifier
- Resolution: Image dimensions (WxH)
- NPU Models Load Time (s): Time to load NPU-accelerated models
- All Models Load Time (s): Total model loading time
- 1st Gen Pipeline Time (s): Warm-up/first generation time
- Avg Pipeline Time (s): Average time excluding first generation
- NPU Memory Usage (MB): Memory used by NPU models
- Total Memory Usage (MB): Total memory footprint
- Inference Steps: Number of denoising steps
"""

import re
import csv
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def parse_benchmark_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parse a benchmark results file and extract pipeline metrics.
    
    Args:
        filepath: Path to the benchmark results text file
        
    Returns:
        List of dictionaries containing parsed metrics for each pipeline
    """
    results = []
    current_pipeline = {}
    in_timing_section = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect pipeline header (e.g., "1. SD 1.5 Pipeline (run_sd.py) with runwayml/stable-diffusion-v1-5 @ 512x512")
        # Pattern: number. Name (script) with model_id @ resolution
        pipeline_match = re.match(r'^\d+\.\s+(.+?)\s+\((.+?)\)\s+with\s+(.+?)\s+@\s+(\d+)x(\d+)', line)
        if pipeline_match:
            # Save previous pipeline if exists
            if current_pipeline:
                results.append(current_pipeline)
            
            # Extract pipeline details
            pipeline_name = pipeline_match.group(1).strip()
            script_name = pipeline_match.group(2).strip()
            model_id = pipeline_match.group(3).strip()
            width = pipeline_match.group(4)
            height = pipeline_match.group(5)
            resolution = f"{width}x{height}"
            
            # Start new pipeline
            current_pipeline = {
                'model_name': pipeline_name,
                'resolution': resolution,
                'npu_load_time': None,
                'all_models_load_time': None,
                'first_gen_time': None,
                'avg_pipeline_time': None,
                'npu_memory': None,
                'total_memory': None,
                'inference_steps': None
            }
            in_timing_section = False
        
        # Extract inference steps - pattern: "   Inference Steps: 20"
        elif line.startswith("Inference Steps:") and current_pipeline:
            match = re.search(r'Inference Steps:\s*(\d+)', line)
            if match:
                current_pipeline['inference_steps'] = int(match.group(1))
        
        # Detect timing section start (appears after Status line)
        elif line.startswith("Timing Summary:"):
            in_timing_section = True
        
        # Parse timing lines
        elif in_timing_section and current_pipeline:
            # NPU models load time - pattern: "Load time of all NPU models: 2.447241s"
            if "load time of all npu models" in line.lower():
                match = re.search(r':\s*([\d.]+)s\s*$', line)
                if match:
                    current_pipeline['npu_load_time'] = float(match.group(1))
            
            # All models load time - pattern: "Load time of all models: 2.447584s"
            elif "load time of all models" in line.lower() and "npu" not in line.lower():
                match = re.search(r':\s*([\d.]+)s\s*$', line)
                if match and current_pipeline['all_models_load_time'] is None:
                    # Only capture the first occurrence (sometimes there are duplicates)
                    current_pipeline['all_models_load_time'] = float(match.group(1))
            
            # First generation time - pattern: "Pipeline time for 1st Gen : 6.139527s"
            elif "pipeline time for 1st gen" in line.lower():
                match = re.search(r':\s*([\d.]+)s\s*$', line)
                if match:
                    current_pipeline['first_gen_time'] = float(match.group(1))
            
            # Average pipeline time - pattern: "Average pipeline time(excluding first iter) : 6.059434s"
            elif "average pipeline time" in line.lower():
                match = re.search(r':\s*([\d.]+)s\s*$', line)
                if match:
                    current_pipeline['avg_pipeline_time'] = float(match.group(1))
            
            # Total NPU memory usage - pattern: "Total NPU memory usage: 1541.12MB (1.51GB)"
            elif "total npu memory usage" in line.lower():
                match = re.search(r':\s*([\d.]+)MB', line, re.IGNORECASE)
                if match:
                    current_pipeline['npu_memory'] = float(match.group(1))
            
            # Total memory usage - pattern: "Total memory usage : 4424.01MB (4.32GB)"
            elif "total memory usage" in line.lower() and "npu" not in line.lower():
                match = re.search(r':\s*([\d.]+)MB', line, re.IGNORECASE)
                if match:
                    current_pipeline['total_memory'] = float(match.group(1))
        
        i += 1
    
    # Don't forget the last pipeline
    if current_pipeline:
        results.append(current_pipeline)
    
    return results


def write_csv(results: List[Dict[str, Any]], output_path: Path):
    """
    Write parsed results to a CSV file.
    
    Args:
        results: List of pipeline metric dictionaries
        output_path: Path where CSV file should be written
    """
    if not results:
        print("⚠️  No results to write")
        return
    
    # Define CSV columns
    fieldnames = [
        'Model Name',
        'Resolution',
        'Inference Steps',
        'NPU Models Load Time (s)',
        'All Models Load Time (s)',
        '1st Gen Pipeline Time (s)',
        'Avg Pipeline Time (s)',
        'NPU Memory Usage (MB)',
        'Total Memory Usage (MB)'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'Model Name': result.get('model_name', 'Unknown'),
                'Resolution': result.get('resolution', 'N/A'),
                'Inference Steps': result.get('inference_steps', ''),
                'NPU Models Load Time (s)': result.get('npu_load_time', ''),
                'All Models Load Time (s)': result.get('all_models_load_time', ''),
                '1st Gen Pipeline Time (s)': result.get('first_gen_time', ''),
                'Avg Pipeline Time (s)': result.get('avg_pipeline_time', ''),
                'NPU Memory Usage (MB)': result.get('npu_memory', ''),
                'Total Memory Usage (MB)': result.get('total_memory', '')
            }
            writer.writerow(row)
    
    print(f"✅ CSV written to: {output_path}")
    print(f"   {len(results)} pipeline(s) processed")


def main():
    """Main entry point for the benchmark post-processor."""
    parser = argparse.ArgumentParser(
        description="Post-process benchmark results and generate CSV summary"
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to specific benchmark results file (default: latest in results/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV filename (default: benchmark_summary_TIMESTAMP.csv)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing benchmark result files (default: results/)'
    )
    
    args = parser.parse_args()
    
    # Determine workspace root (parent of scripts directory)
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    results_dir = workspace_root / args.results_dir
    
    # Find input file
    if args.input:
        input_file = Path(args.input)
        if not input_file.is_absolute():
            input_file = workspace_root / input_file
    else:
        # Find latest benchmark result file
        benchmark_files = sorted(results_dir.glob("pipeline_results_benchmark_*.txt"))
        if not benchmark_files:
            print(f"❌ No benchmark result files found in {results_dir}")
            print(f"   Looking for files matching: pipeline_results_benchmark_*.txt")
            return 1
        input_file = benchmark_files[-1]
        print(f"📄 Processing latest benchmark file: {input_file.name}")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    # Parse the results
    print(f"🔍 Parsing benchmark results...")
    results = parse_benchmark_file(input_file)
    
    if not results:
        print("❌ No pipeline results found in the file")
        return 1
    
    print(f"✅ Found {len(results)} pipeline result(s)")
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
        if not output_file.is_absolute():
            output_file = results_dir / output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"benchmark_summary_{timestamp}.csv"
    
    # Write CSV
    print(f"📊 Generating CSV summary...")
    write_csv(results, output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())
