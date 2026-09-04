"""List the 43 E-GMD kits with hours and a rough acoustic/electronic tag.

Use the output to fill configs/kits_v1.yaml.

    python scripts/list_kits.py [--csv data/e-gmd-v1.0.0.csv]
"""
import argparse
import pandas as pd

ELECTRONIC_HINTS = ("808", "909", "electr", "dance", "house", "techno",
                    "hip", "edm", "synth", "drum machine", "lo-fi", "lofi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/e-gmd-v1.0.0.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["audio_filename"].notna()]
    g = (df.groupby("kit_name")
           .agg(hours=("duration", lambda s: s.sum() / 3600),
                sequences=("id", "nunique"))
           .sort_values("hours", ascending=False))
    g["tag_guess"] = ["electronic?" if any(h in k.lower() for h in ELECTRONIC_HINTS)
                      else "acoustic?" for k in g.index]

    pd.set_option("display.width", 120)
    print(g.to_string(float_format=lambda x: f"{x:5.1f}"))
    print(f"\n{len(g)} kits, {g['hours'].sum():.1f} audio hours total")
    print("Edit configs/kits_v1.yaml with 6 train kits (spread acoustic->electronic) "
          "and 8 held-out kits.")


if __name__ == "__main__":
    main()
