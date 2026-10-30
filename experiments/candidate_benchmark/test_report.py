"""Regression for numeric gallery keys and completeness of measured report tables."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from run import read_rows


def test(folder):
    construction=read_rows(folder/"construction.csv")
    summary=read_rows(folder/"summary.csv")
    text=(folder/"REPORT.md").read_text(encoding="utf-8")
    for g in (16,128,1024):
        available=any(r["phase"]=="test" and r["suite"]=="scaled" and int(r["gallery"])==g and r["profile"]=="dense_target" and int(r["length"])==32768 for r in construction)
        if available:
            assert text.count(f"| {g} | ")>=2, "missing performance/memory table rows (numeric CSV keys)"
    assert len(summary)==len(construction)*50
    assert all(float(r["candidate_recall"])<=.5 for r in summary if r["budget"]=="1")
    assert "occasional gains do not establish a reliable benefit" in text
    assert "CPU exact-index bytes are an allocation estimate" in text
    print("PASS: report performance/memory rows, complete strategy/budget matrix, recall denominator and limitations")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("folder",type=Path); test(p.parse_args().folder)
