"""Apply the source adaptations documented in install/INSTALLATION.md before simulator builds."""
import argparse
import os
from pathlib import Path
import re


CMAKE_MINIMUM = re.compile(
    r'(cmake_minimum_required\s*\(\s*VERSION\s+)(\d+(?:\.\d+)*)(?=[\s.)])',
    re.IGNORECASE,
)


def patch_text(path, source, mode):
    if path.name == 'CMakeLists.txt' or path.suffix == '.cmake':
        def raise_minimum(match):
            version = tuple(int(part) for part in match[2].split('.'))
            return match[1] + ('3.5' if (version + (0, 0))[:2] < (3, 5) else match[2])
        return CMAKE_MINIMUM.sub(raise_minimum, source)
    if mode == 'mattersim' and path.suffix in {'.h', '.hpp', '.cpp', '.cc', '.cxx'}:
        source = re.sub(r'\bCV_LOAD_IMAGE_ANYDEPTH\b', 'cv::IMREAD_ANYDEPTH', source)
        source = re.sub(r'(?<![\w:])NORM_L2\b', 'cv::NORM_L2', source)
        # Change both the buffer bound and ss_size; never touch system headers.
        if path.name.lower() in {'catch.hpp', 'catch_all.hpp'}:
            source = re.sub(r'\bSIGSTKSZ\b', '32768', source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['habitat-sim', 'mattersim'])
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    root = checkout / 'src' if args.mode == 'habitat-sim' else checkout
    if not root.is_dir() or not (root / 'CMakeLists.txt').is_file():
        parser.error(f'Expected simulator source tree with CMakeLists.txt: {root}')
    changed = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', 'build', '__pycache__'}
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or not (name == 'CMakeLists.txt' or path.suffix in
                    {'.cmake', '.h', '.hpp', '.cpp', '.cc', '.cxx'}):
                continue
            original = path.read_bytes()
            source = original.decode('utf-8')
            patched = patch_text(path, source, args.mode).encode('utf-8')
            if patched != original:
                path.write_bytes(patched)
                changed += 1
                print(f'Patched {path.relative_to(checkout)}')
    print(f'{changed} files changed. Review the source diff before compiling.')


if __name__ == '__main__':
    main()
