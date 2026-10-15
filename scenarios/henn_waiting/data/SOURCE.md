# Data: the converted Henn instances (Heßler-Irnich format)

Sixteen instances in the Heßler-Irnich text format (`<n>s-<orders>-<capacity>-0.txt`)
and their arrival times (`HennOnline/Arrival_Time_abc1_<stem>.csv`, milliseconds).
They were added to this repository in commit `107ff2d` (28 April 2026) and held
under `scenarios/scenario_henn_online/data/` until Phase S. Their item positions
differ from the published OOBP archive; only the arrival files are identical
(F60, `docs/oobp_source_comparison.md`), so results on them are not comparable
with published OOBP values. No public URL or licence is known.

Restore them from the repository history and check them:

```bash
python scripts/fetch_data.py henn_waiting
```

`SHA256SUMS.txt` lists every file.
