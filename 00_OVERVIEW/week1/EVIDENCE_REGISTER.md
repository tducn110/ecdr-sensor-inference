# Week 1 evidence register

| ID | Requirement / claim | Status | Scope / artifact |
|---|---|---|---|
| W1-01 | Project brief and research grounding | VERIFIED | Canonical brief, bibliography and targeted comparison |
| W1-02 | K=12 lock | VERIFIED RUN | Selected seed 43, epoch 8; checkpoint and preprocessing identities registered |
| W1-03 | Exact architecture | VERIFIED RUN | Instantiated graph and parameter-count/shape checks passed |
| W1-04 | Full reference pipeline executed | VERIFIED RUN | Run `week1_ref_1793620216_9d6c1d85`; all phases exit 0; pipeline hash `1ef93101f658f8a18ebf19f0c8743ec5311ffe94933aee3505be6ec29a0be4d6` |
| W1-05 | Transition label policy/support blocker | VERIFIED SNAPSHOT | 344 pure transition vs 10,017 pure basic; G2 HOLD is caused by transition undersupport |
| W1-06 | Main RQ and objectives | VERIFIED | Accuracy, reference agreement, net cost separated |
| W1-07 | Evidence register claim status | VERIFIED | This register plus native PDF table; independent baseline receipt passes 65/65 |
| W1-08 | Sensor/data full contract | VERIFIED AUDIT PASS | 17,453 windows, 61 recordings, shape 17,453x6x128 float32; 131 source/data hashes and 122 raw sensor files |
| W1-09 | Hypothesis ledger | MEASURED / PROPOSED / DERIVED | H1 descriptively supported; H2/H4 future; H3 derived |
| W1-10 | Stable/transition stability plus future coverage/fallback/cost strata | VERIFIED DESCRIPTIVE / PROPOSED | 5,034 valid adjacent pairs; future pure-basic/pure-transition/mixed matched-cost evaluation; no independent-population claim |
| W1-11 | Week roadmap | VERIFIED | Week 1 baseline, Week 2 data/model rationale, Week 3 certification |
| W1-12 | English presentation | VERIFIED | Editable LaTeX body, wrapper and companion records |

Independent baseline artifact verification passed 65/65 checks with zero
failures at `output/week1_closeout_20261102/reference/independent_verification.json`.
The finite batch-1 versus batch-512 check matched labels for all 5,053 windows
with zero disagreements and maximum absolute logit difference
9.5367431640625e-6. This is baseline consistency evidence, not formal
certificate verification.

Reference timing currently means preprocessing plus reference forward evaluation
only; it excludes raw acquisition, windowing, event detection, certificate and
membership work, and I/O. It is a baseline and does not claim full ECDR
end-to-end savings. Aggregate three-seed spread is population standard
deviation (ddof=0), not a confidence interval. Any Wilson per-class recall
intervals are nominal descriptive intervals because overlapping windows are not
independent samples; no sample-independence claim is made. A finite CPU batch-1
versus batch-512 prediction-consistency check completed for the recorded run is
evidence for that tested configuration only, not a universal numerical proof.
Any future metric must include run ID, script/command, environment versions,
checkpoint/model/preprocessing/data hashes, timing scope and denominator.

The live audit artifact reports zero raw-slice mismatches, zero label-builder
mismatches, and verified aligned overlap for all pairs. A one-sample offset
changes 98 window policy/label assignments; this is an explicit provenance
caveat, not a mutation of the current run. Pure support is 10,017 basic and
344 transition windows. The support table reports every split's window and
subject support for pure and pure+majority policies; all training majority
classes reach 50, but six transition validation cells and four transition test
cells remain below 50. The pure policy has 17 low-support cells, including the
class-8 validation window-and-subject criterion; the documented blocker count
is 18 criteria when both dimensions are counted. Run status is audit pass, not
a quality-gate pass.
