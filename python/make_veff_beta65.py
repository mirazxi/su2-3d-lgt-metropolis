from pathlib import Path
from analyze_scaling_smear import read_wilson_csv, build_veff, save_veff_csv

cases = [
    Path("runs/scaling_L20_beta_6p5"),
    Path("runs/scaling_L22_beta_6p5"),
]

for case_dir in cases:
    print(f"Processing {case_dir}")

    ape_data = read_wilson_csv(case_dir / "wilson_loops_ape.csv")
    veff_ape = build_veff(ape_data)
    save_veff_csv(case_dir / "veff_ape.csv", veff_ape)

    raw_data = read_wilson_csv(case_dir / "wilson_loops_raw.csv")
    veff_raw = build_veff(raw_data)
    save_veff_csv(case_dir / "veff_raw.csv", veff_raw)

    print(f"Wrote {case_dir / 'veff_ape.csv'}")
    print(f"Wrote {case_dir / 'veff_raw.csv'}")
