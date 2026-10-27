# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Sync-check test: VALID_ARGS (config_validator.py) vs runner_args.py parser.

PURPOSE: The config_validator maintains a static VALID_ARGS dictionary that
mirrors the argparse parser in src/utils/runner_args.py. If someone adds a new
argument to the parser but forgets to update VALID_ARGS, the validator will
produce false-positive "unknown argument" warnings. This test catches that drift
immediately by comparing both registries and failing on any mismatch.

Run: python scripts/test_arg_sync.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so both src.utils and scripts are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def get_parser_flags():
    """Extract all flag names from the argparse parser in runner_args.py."""
    from src.utils.runner_args import parser

    # Built-in argparse flags that don't need validation
    SKIP_FLAGS = {"-h", "--help"}

    flags = set()
    for action in parser._actions:
        if action.option_strings:
            for flag in action.option_strings:
                if flag not in SKIP_FLAGS:
                    flags.add(flag)
    return flags


def get_validator_flags():
    """Extract all flag names from config_validator.VALID_ARGS."""
    from config_validator import VALID_ARGS

    return set(VALID_ARGS.keys())


def test_all_parser_flags_in_validator():
    """Every flag in runner_args.py parser must appear in VALID_ARGS."""
    parser_flags = get_parser_flags()
    validator_flags = get_validator_flags()

    missing = parser_flags - validator_flags
    if missing:
        print(f"FAIL: Flags in runner_args.py but missing from VALID_ARGS:")
        for f in sorted(missing):
            print(f"  {f}")
        return False

    print(f"PASS: All {len(parser_flags)} parser flags present in VALID_ARGS")
    return True


def test_no_stale_validator_flags():
    """Every flag in VALID_ARGS should exist in runner_args.py parser."""
    parser_flags = get_parser_flags()
    validator_flags = get_validator_flags()

    stale = validator_flags - parser_flags
    if stale:
        print(f"FAIL: Flags in VALID_ARGS but not in runner_args.py parser (stale?):")
        for f in sorted(stale):
            print(f"  {f}")
        return False

    print(f"PASS: No stale flags in VALID_ARGS")
    return True


def test_type_consistency():
    """Verify VALID_ARGS type annotations match the argparse action/type."""
    from src.utils.runner_args import parser
    from config_validator import VALID_ARGS

    # Build a map: flag -> argparse action
    flag_to_action = {}
    for action in parser._actions:
        for flag in action.option_strings:
            flag_to_action[flag] = action

    errors = []
    for flag, spec in VALID_ARGS.items():
        if flag not in flag_to_action:
            continue  # caught by other tests

        action = flag_to_action[flag]
        expected_type = spec["type"]

        # store_true -> bool
        if action.const is True and action.nargs == 0:
            if expected_type != "bool":
                errors.append(f"{flag}: parser is store_true but VALID_ARGS says '{expected_type}'")
        # nargs="+" -> list type
        elif action.nargs in ("+", "*"):
            if not expected_type.endswith("_list"):
                errors.append(f"{flag}: parser has nargs='{action.nargs}' but VALID_ARGS says '{expected_type}' (expected *_list)")

    if errors:
        print(f"FAIL: Type mismatches between parser and VALID_ARGS:")
        for e in errors:
            print(f"  {e}")
        return False

    print(f"PASS: Type annotations consistent")
    return True


if __name__ == "__main__":
    results = [
        test_all_parser_flags_in_validator(),
        test_no_stale_validator_flags(),
        test_type_consistency(),
    ]

    print()
    if all(results):
        print("All sync checks passed.")
        sys.exit(0)
    else:
        print("Sync check FAILED -- update VALID_ARGS in config_validator.py")
        sys.exit(1)
