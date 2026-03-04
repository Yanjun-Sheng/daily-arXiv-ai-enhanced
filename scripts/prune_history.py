from pathlib import Path
import re


def main() -> None:
    retention_days = 14
    root = Path('data')
    root.mkdir(exist_ok=True)

    # Collect dates that have AI-enhanced files (these are what frontend renders)
    enhanced_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})_AI_enhanced_(English|Chinese)\.jsonl$')
    dates = set()
    for fp in root.glob('*_AI_enhanced_*.jsonl'):
        m = enhanced_pattern.match(fp.name)
        if m:
            dates.add(m.group(1))

    # Fallback for early runs: if no enhanced files yet, use raw daily snapshots
    if not dates:
        raw_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})\.jsonl$')
        for fp in root.glob('*.jsonl'):
            m = raw_pattern.match(fp.name)
            if m:
                dates.add(m.group(1))

    sorted_dates = sorted(dates, reverse=True)
    keep_dates = set(sorted_dates[:retention_days])

    for fp in root.glob('*'):
        m = re.match(r'^(\d{4}-\d{2}-\d{2})', fp.name)
        if not m:
            continue
        if m.group(1) not in keep_dates:
            fp.unlink(missing_ok=True)

    print(f"kept_dates={sorted(keep_dates, reverse=True)}")


if __name__ == '__main__':
    main()
