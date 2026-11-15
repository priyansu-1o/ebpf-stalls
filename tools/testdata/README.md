# Unit Test Fixture Data (tools/testdata/)

THIS DIRECTORY CONTAINS SYNTHETIC UNIT-TEST FIXTURES FOR CODE TESTING ONLY.
IT IS NOT EVALUATION DATA, WILL NEVER BE PLACED UNDER data/, AND MUST NEVER
BE INCLUDED IN ANY BENCHMARK METRIC OR EVALUATION REPORT.

Contents:
- `truth.jsonl`: 3 operations designed with known hand-calculated bounds.
  - Op 1: D-state blocked sleep (prev_state = 2) with wakeup and switch_in
  - Op 2: Interruptible sleep (prev_state = 1) with wakeup and switch_in
  - Op 3: Preempted runnable wait (prev_state = 0) with switch_in
- `events.csv`: Minimal hand-crafted trace matching the schema with exact timestamps.

