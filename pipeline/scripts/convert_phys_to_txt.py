from __future__ import annotations

import argparse
from pathlib import Path

from astropy import units as u


def convert_phys_file(input_file: str | Path, output_file: str | Path) -> None:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_file, "r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            rows.append(
                {
                    "time": values[0],
                    "x_au": float(values[1]),
                    "y_au": float(values[2]),
                    "z_au": float(values[3]),
                    "density": values[4],
                    "temp": values[5],
                    "av": values[6],
                }
            )

    with open(output_file, "w", encoding="ascii") as handle:
        for row in rows:
            x_pc = (row["x_au"] * u.AU).to(u.pc).value
            y_pc = (row["y_au"] * u.AU).to(u.pc).value
            z_pc = (row["z_au"] * u.AU).to(u.pc).value
            handle.write(
                f"    {x_pc:.14e}"
                f"    {y_pc:.14e}"
                f"    {z_pc:.14e}"
                f"    {row['density']}"
                f"    {row['temp']}"
                f"    {row['temp']}"
                f"    {row['av']}"
                f"    0.00000e+00"
                f"    0.00000e+00"
                f"    {row['time']}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PHANTOM .phys trace file to chemistry-model input text.")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_phys_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
