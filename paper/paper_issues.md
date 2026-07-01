# Issues Found in Current LaTeX

## Issue 1: DUPLICATE TABLE (Critical)
**Table II** (`tab:models`, Section III-D) and **Table IV** (`tab:perf`, Section IV-A) contain **IDENTICAL data**.
Both show the exact same 6 models with the same Accuracy, F1, ROC-AUC, PR-AUC, and Threshold values.

**Fix**: Remove Table IV (`tab:perf`) entirely from Section IV-A. Keep only Table II (`tab:models`) in Section III-D.
Update the text in Section IV-A to reference `Table~\ref{tab:models}` instead of `Table~\ref{tab:perf}`.

## Issue 2: Figure comment numbers are wrong
The LaTeX comments say "Fig. 3", "Fig. 2", etc. but LaTeX numbers figures SEQUENTIALLY by order of appearance.

Actual numbering will be:
- Fig. 1 = flowchart.png (Section III-F) ✅
- Fig. 2 = through(Reno).png (after Reno results)
- Fig. 3 = overall.png (after Cubic results)
- Fig. 4 = cwnd_trace(Reno).png (after ECN signal analysis)
- Fig. 5 = cwnd_trace(cubic).png (after Fig. 4)

## Issue 3: Figure placement could be better
- `through(Reno).png` (Reno throughput) is placed correctly after Reno discussion ✅
- `overall.png` is placed after Cubic but should arguably go in Comparative Summary (IV-E)
- Both CWND traces are placed after ECN Signal Analysis — would be better if:
  - Reno CWND trace → in Section IV-B (Reno results)
  - Cubic CWND trace → in Section IV-C (Cubic results)

## Recommended figure order (matches paper flow):
1. Fig. 1 — flowchart.png → Section III-F ✅
2. Fig. 2 — cwnd_trace(Reno).png → Section IV-B (after Reno table/discussion)
3. Fig. 3 — through(Reno).png → Section IV-B (after Fig. 2)
4. Fig. 4 — cwnd_trace(cubic).png → Section IV-C (after Cubic table/discussion)
5. Fig. 5 — overall.png → Section IV-E (Comparative Summary)
