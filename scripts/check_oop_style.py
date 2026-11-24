#!/usr/bin/env python3
"""Статический анализатор строгого ООП-стиля исходного кода."""

import ast
import sys
from pathlib import Path

ALLOWED_MAIN_FILES = frozenset({"cli.py"})


class OOPStyleChecker:
    """Проверяет отсутствие процедурных функций верхнего уровня в модулях src."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    @classmethod
    def _has_main_guard(cls, tree: ast.Module) -> bool:
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            src_seg = ast.unparse(node.test)
            if "__name__" in src_seg and "__main__" in src_seg:
                return True
        return False

    @classmethod
    def _is_allowed_top_level_func(
        cls,
        file_path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        has_main_guard: bool,
    ) -> bool:
        return bool(file_path.name in ALLOWED_MAIN_FILES and node.name == "main" and has_main_guard)

    def check_file(self, file_path: Path) -> None:
        """Проверяет один Python-файл на соответствие ООП-стандарту."""
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        has_main_guard = self._has_main_guard(tree)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not self._is_allowed_top_level_func(
                file_path, node, has_main_guard
            ):
                self.violations.append(
                    f"{file_path}:{node.lineno}: запрещена процедурная функция '{node.name}' вне класса (используй ООП-класс/метод)"
                )

    def check_paths(self, paths: list[str]) -> int:
        """Запускает проверку всех Python-файлов в указанных путях."""
        target_files: list[Path] = []
        for raw in paths or ["src"]:
            p = Path(raw)
            if p.is_file() and p.suffix == ".py":
                target_files.append(p)
            elif p.is_dir():
                target_files.extend(sorted(p.rglob("*.py")))

        checked = 0
        for fp in target_files:
            if ".venv" in fp.parts:
                continue
            checked += 1
            self.check_file(fp)

        for msg in self.violations:
            print(f"❌ OOP VIOLATION: {msg}")
        print(f"\n[OOP Style Check] Verified {checked} files: {len(self.violations)} violations.")
        return 1 if self.violations else 0


if __name__ == "__main__":
    sys.exit(OOPStyleChecker().check_paths(sys.argv[1:]))
