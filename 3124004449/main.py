#!/usr/bin/env python3
"""
Paper Plagiarism Checker - Main Entry Point

Usage:
    python main.py <original_file> <plagiarized_file> <output_file>

Example:
    python main.py orig.txt plagiarized.txt answer.txt

The program reads the original and plagiarized texts, computes the
plagiarism rate, and writes the result (a float between 0.00 and 1.00
with 2 decimal places) to the output file.
"""

import sys
import os
import traceback

from checker.algorithm import calculate_similarity


def read_file(file_path: str) -> str:
    """Read text file content with automatic encoding detection.

    Tries UTF-8 first, then falls back to common Chinese encodings.

    Args:
        file_path: Absolute or relative path to the file

    Returns:
        File content as a string

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be decoded with any supported encoding
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    # Check BOM-marked files before legacy encodings.  GBK can decode some
    # UTF-16 byte sequences without raising an exception, producing garbage.
    encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"Cannot decode file with any supported encoding: {file_path}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 4:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<original_file> <plagiarized_file> <output_file>",
              file=sys.stderr)
        sys.exit(1)

    orig_path = sys.argv[1]
    plag_path = sys.argv[2]
    out_path = sys.argv[3]

    # Read input files
    try:
        orig_text = read_file(orig_path)
        plag_text = read_file(plag_path)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate similarity
    try:
        similarity = calculate_similarity(orig_text, plag_text)
    except (TypeError, ValueError, RuntimeError) as e:
        print(f"Error calculating similarity: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Clamp and format
    similarity = max(0.0, min(1.0, similarity))

    # Write output
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
    except IOError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
