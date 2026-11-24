#!/usr/bin/env python3
"""Скрипт проверки лимита строк в файлах исходного кода (Code Metric Gatekeeper).

Запускается локально и в pre-commit.
Предупреждает при превышении 300 строк, завершается ошибкой при превышении 500 строк.
"""

import sys
from pathlib import Path

WARNING_THRESHOLD = 300
ERROR_THRESHOLD = 500


def check_files(paths: list[str]) -> int:
    exit_code = 0
    checked_files = 0
    warnings_count = 0
    errors_count = 0

    target_paths: list[Path] = []
    if paths:
        for p_str in paths:
            p = Path(p_str)
            if p.is_file() and p.suffix == ".py":
                target_paths.append(p)
            elif p.is_dir():
                target_paths.extend(p.rglob("*.py"))
    else:
        target_paths = list(Path("src").rglob("*.py"))

    for file_path in target_paths:
        if ".venv" in file_path.parts:
            continue

        try:
            lines = len(file_path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Skipping {file_path}: {exc}", file=sys.stderr)
            continue

        checked_files += 1
        if lines > ERROR_THRESHOLD:
            print(f"❌ ERROR: {file_path} exceeds {ERROR_THRESHOLD} lines: {lines} lines!")
            errors_count += 1
            exit_code = 1
        elif lines > WARNING_THRESHOLD:
            print(f"⚠️  WARNING: {file_path} is long ({lines} lines > {WARNING_THRESHOLD})")
            warnings_count += 1

    print(f"\n[File Length Check] Verified {checked_files} files: {errors_count} errors, {warnings_count} warnings.")
    return exit_code


if __name__ == "__main__":
    sys.exit(check_files(sys.argv[1:]))
