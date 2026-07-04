"""
Aggregate OpenEMMA benchmark CSV results without pandas.

Writes one condition-level summary CSV next to the benchmark output and prints
condition, town, and weather rollups to stdout.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


METRICS = [
    'completion_pct',
    'distance_m',
    'offroad_pct',
    'collisions',
    'stuck_events',
    'vlm_frame_pct',
    'fallback_frame_pct',
    'speed_floor_pct',
    'hallucination_pct',
    'avg_speed_mps',
    'steer_std',
]


def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number):
        return None
    return number


def load_rows(csv_path):
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def group_rows(rows, key_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, '') for field in key_fields)
        groups[key].append(row)
    return groups


def metric_values(rows, metric):
    values = []
    for row in rows:
        value = parse_number(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def compute_stats(rows):
    stats = {}
    for metric in METRICS:
        values = metric_values(rows, metric)
        if values:
            array = np.asarray(values, dtype=float)
            stats[f'{metric}_mean'] = float(np.mean(array))
            stats[f'{metric}_std'] = float(np.std(array))
        else:
            stats[f'{metric}_mean'] = ''
            stats[f'{metric}_std'] = ''
    return stats


def compute_means(rows):
    means = {}
    for metric in METRICS:
        values = metric_values(rows, metric)
        if values:
            means[metric] = float(np.mean(np.asarray(values, dtype=float)))
        else:
            means[metric] = ''
    return means


def format_number(value):
    if value == '':
        return 'n/a'
    return f'{float(value):.2f}'


def format_mean_std(row, metric):
    mean = row.get(f'{metric}_mean', '')
    std = row.get(f'{metric}_std', '')
    if mean == '' or std == '':
        return 'n/a'
    return f'{float(mean):.2f}+/-{float(std):.2f}'


def print_table(title, headers, rows):
    print()
    print(title)
    if not rows:
        print('  no rows')
        return

    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]
    header_line = '  '.join(
        str(header).ljust(widths[index])
        for index, header in enumerate(headers)
    )
    separator = '  '.join('-' * width for width in widths)
    print(header_line)
    print(separator)
    for row in string_rows:
        print('  '.join(
            row[index].ljust(widths[index])
            for index in range(len(headers))
        ))


def write_summary_csv(summary_path, condition_rows):
    parent = os.path.dirname(os.path.abspath(summary_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = ['town', 'weather_label', 'n']
    for metric in METRICS:
        fieldnames.extend([f'{metric}_mean', f'{metric}_std'])

    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in condition_rows:
            output = {}
            for field in fieldnames:
                value = row.get(field, '')
                if isinstance(value, float):
                    value = round(value, 6)
                output[field] = value
            writer.writerow(output)


def aggregate_results(csv_path, summary_path=None):
    rows = load_rows(csv_path)
    if summary_path is None:
        summary_path = os.path.join(
            os.path.dirname(os.path.abspath(csv_path)),
            'summary.csv',
        )

    condition_rows = []
    for (town, weather_label), grouped in sorted(
        group_rows(rows, ('town', 'weather_label')).items()
    ):
        condition_rows.append({
            'town': town,
            'weather_label': weather_label,
            'n': len(grouped),
            **compute_stats(grouped),
        })

    write_summary_csv(summary_path, condition_rows)

    condition_table_rows = []
    for row in condition_rows:
        condition_table_rows.append(
            [row['town'], row['weather_label'], row['n']]
            + [format_mean_std(row, metric) for metric in METRICS]
        )
    print_table(
        'Condition summary (mean+/-std)',
        ['town', 'weather', 'n'] + METRICS,
        condition_table_rows,
    )

    town_table_rows = []
    for (town,), grouped in sorted(group_rows(rows, ('town',)).items()):
        means = compute_means(grouped)
        town_table_rows.append(
            [town, len(grouped)] + [format_number(means[metric]) for metric in METRICS]
        )
    print_table(
        'Town rollup (mean)',
        ['town', 'n'] + METRICS,
        town_table_rows,
    )

    weather_table_rows = []
    for (weather_label,), grouped in sorted(group_rows(rows, ('weather_label',)).items()):
        means = compute_means(grouped)
        weather_table_rows.append(
            [weather_label, len(grouped)]
            + [format_number(means[metric]) for metric in METRICS]
        )
    print_table(
        'Weather rollup (mean)',
        ['weather', 'n'] + METRICS,
        weather_table_rows,
    )

    print()
    print(f'[Aggregate] Wrote {summary_path}')
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate OpenEMMA benchmark CSV results'
    )
    parser.add_argument('csv_path', help='Benchmark results CSV')
    parser.add_argument('--summary-out', default=None,
                        help='Optional summary CSV output path')
    return parser.parse_args()


def main():
    args = parse_args()
    aggregate_results(args.csv_path, args.summary_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
