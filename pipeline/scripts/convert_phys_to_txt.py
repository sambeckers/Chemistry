from __future__ import annotations

import argparse
from pathlib import Path

from astropy import units as u


class PhysTraceConverter:
    """Convert PHANTOM .phys trace files into chemistry model text inputs."""

    @staticmethod
    def _parse_phys_lines(lines: list[str]) -> list[dict[str, str | float]]:
        rows: list[dict[str, str | float]] = []
        for line in lines:
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
        return rows

    @staticmethod
    def _write_model_input(rows: list[dict[str, str | float]], output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="ascii") as handle:
            for row in rows:
                # Model input expects parsec positions while PHANTOM traces use AU.
                x_pc = (float(row["x_au"]) * u.AU).to(u.pc).value
                y_pc = (float(row["y_au"]) * u.AU).to(u.pc).value
                z_pc = (float(row["z_au"]) * u.AU).to(u.pc).value
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

    @classmethod
    def convert_phys_file(cls, input_file: str | Path, output_file: str | Path) -> None:
        """Read one .phys file and write one model-ready plain text file."""
        input_file = Path(input_file)
        rows = cls._parse_phys_lines(input_file.read_text(encoding="ascii").splitlines())
        cls._write_model_input(rows, Path(output_file))

    @classmethod
    def convert_phys_text(cls, text: str, output_file: str | Path) -> None:
        """Convert .phys text content directly without requiring an on-disk .phys file."""
        rows = cls._parse_phys_lines(text.splitlines())
        cls._write_model_input(rows, Path(output_file))


class PhysTraceCLI:
    """CLI wrapper for file conversion."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line input and output path arguments."""
        parser = argparse.ArgumentParser(description="Convert a PHANTOM .phys trace file to chemistry-model input text.")
        parser.add_argument("input_file")
        parser.add_argument("output_file")
        return parser.parse_args()

    @classmethod
    def run(cls) -> None:
        """Run conversion from parsed arguments."""
        args = cls.parse_args()
        PhysTraceConverter.convert_phys_file(args.input_file, args.output_file)


def main() -> None:
    """CLI entrypoint."""
    PhysTraceCLI.run()


if __name__ == "__main__":
    main()
