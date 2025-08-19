#!/usr/bin/env python3
"""
Examples of how to use the enhanced filtering system in comprehensive_correction_workflow.py
"""

def print_examples():
    print("🔍 FILTERING EXAMPLES for comprehensive_correction_workflow.py")
    print("=" * 80)
    
    print("\n📋 BASIC FILTERING:")
    print("--filter_config fiddlercrab_corneas     # Filter by config name")
    print("--filter_scan dampieri                  # Filter by scan name")
    print("--filter_species paraphronima           # Filter by species")
    print("--filter_structure corneas              # Filter by structure type")
    print("--min_f1 0.8                           # Only models with F1 >= 0.8")
    print("--max_fps 50                           # Only models with ≤50 false positives")
    
    print("\n🎯 REGEX FILTERING (use --regex flag):")
    print("--filter_config 'fiddlercrab.*corneas' --regex")
    print("--filter_config '(paraphronima|fiddlercrab)_corneas' --regex")
    print("--filter_scan '^dampieri.*16' --regex")
    print("--filter_species '(dampieri|flammula)' --regex")
    
    print("\n🔧 COMPLEX COMBINATIONS:")
    print("# High-performing fiddlercrab cornea models:")
    print("--filter_structure corneas --filter_species fiddlercrab --min_f1 0.9")
    
    print("\n# Models with specific issues:")
    print("--max_fps 10 --max_fns 20  # Models with few errors")
    
    print("\n# Regex: all 'without_' config variants:")
    print("--filter_config 'without_.*' --regex")
    
    print("\n# Regex: specific scan patterns:")
    print("--filter_scan '^P_.*FEG.*' --regex  # P_species with FEG prefix")
    
    print("\n💻 FULL COMMAND EXAMPLES:")
    print("=" * 80)
    
    examples = [
        {
            "desc": "Process only fiddlercrab cornea models",
            "cmd": "python comprehensive_correction_workflow.py --filter_structure corneas --filter_species fiddlercrab"
        },
        {
            "desc": "High F1 score paraphronima models (regex)",
            "cmd": "python comprehensive_correction_workflow.py --filter_species 'paraphronima' --min_f1 0.85 --regex"
        },
        {
            "desc": "Models with preprocessing variants (regex)",
            "cmd": "python comprehensive_correction_workflow.py --filter_config '(hist_std|without_.*|from_pretrained)' --regex"
        },
        {
            "desc": "Specific scan and low error count",
            "cmd": "python comprehensive_correction_workflow.py --filter_scan dampieri --max_fps 5 --max_fns 10"
        },
        {
            "desc": "Review mode with filtering",
            "cmd": "python comprehensive_correction_workflow.py --review --filter_structure rhabdoms --min_f1 0.7"
        },
        {
            "desc": "Models needing correction (medium F1 range)",
            "cmd": "python comprehensive_correction_workflow.py --min_f1 0.6 --max_f1 0.9 --no-template"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}")
        print(f"   {example['cmd']}")
    
    print("\n📊 REGEX PATTERNS REFERENCE:")
    print("=" * 80)
    print("^text        # Starts with 'text'")
    print("text$        # Ends with 'text'")
    print(".*           # Match any characters")
    print("(a|b)        # Match 'a' OR 'b'")
    print("[abc]        # Match any character: a, b, or c")
    print("\\d+          # Match one or more digits")
    print("without_.*   # Match 'without_' followed by anything")
    
    print("\n🏃‍♂️ QUICK START:")
    print("=" * 80)
    print("# See what models are available:")
    print("python comprehensive_correction_workflow.py --filter_structure corneas")
    print("")
    print("# Start with high-performing models:")
    print("python comprehensive_correction_workflow.py --min_f1 0.8")
    print("")
    print("# Work on a specific species:")
    print("python comprehensive_correction_workflow.py --filter_species fiddlercrab")

if __name__ == "__main__":
    print_examples()