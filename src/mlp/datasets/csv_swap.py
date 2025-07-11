
import argparse
import csv
import ast
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Swap specified values in a target column of a CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to the output CSV file. Defaults to overwrite the input file."
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target column (index or header name) to apply swaps."
    )
    parser.add_argument(
        "--swap",
        required=True,
        help=(
            "Tuple of pairs specifying values to swap, "
            "e.g. '((0,1),(1,0))' or \"(('yes','no'),('no','yes'))\"."
        )
    )
    parser.add_argument(
        "--sep", "-s",
        default=",",
        help="CSV delimiter to use when reading and writing (default: ',')."
    )
    return parser.parse_args()

def load_swap_mapping(swap_arg):
    try:
        mapping_raw = ast.literal_eval(swap_arg)
    except (ValueError, SyntaxError):
        sys.exit("Error: --swap argument is not a valid Python literal.")
    if (
        not hasattr(mapping_raw, "__iter__")
        or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            for pair in mapping_raw
        )
    ):
        sys.exit("Error: --swap must be an iterable of 2-item tuples.")
    return dict(mapping_raw)

def apply_swap(value, mapping):
    for key, newval in mapping.items():
        # numeric match
        if isinstance(key, int):
            try:
                if int(value) == key:
                    return str(newval)
            except ValueError:
                continue
        elif isinstance(key, float):
            try:
                if float(value) == key:
                    return str(newval)
            except ValueError:
                continue
        # string match
        elif isinstance(key, str):
            if value == key:
                return str(newval)
    return value

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path
    sep = args.sep

    if not os.path.isfile(input_path):
        sys.exit(f"Error: Input file '{input_path}' not found.")

    swap_mapping = load_swap_mapping(args.swap)
    target_arg = args.target

    # Read the CSV
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter=sep)
        rows = list(reader)

    if not rows:
        sys.exit("Error: Input CSV is empty.")

    header = rows[0]
    data_rows = rows[1:]

    # Determine column index
    if target_arg.isdigit():
        col_index = int(target_arg)
        if col_index < 0 or col_index >= len(header):
            sys.exit(f"Error: Column index {col_index} is out of range.")
    else:
        if target_arg not in header:
            sys.exit(f"Error: Column name '{target_arg}' not found in header.")
        col_index = header.index(target_arg)

    # Process rows
    processed = [header]
    for row_num, row in enumerate(data_rows, start=2):
        if len(row) <= col_index:
            sys.exit(
                f"Error: Row {row_num} has only {len(row)} columns; "
                f"cannot access column {col_index}."
            )
        row[col_index] = apply_swap(row[col_index], swap_mapping)
        processed.append(row)

    # Write out with specified delimiter
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=sep)
        writer.writerows(processed)

    print(f"Swapping complete. Output written to '{output_path}'.")

if __name__ == "__main__":
    main()

