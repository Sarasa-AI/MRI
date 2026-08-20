# Decisions

## D-001 Portfolio ownership

The MRI repo is the portfolio control plane, not a duplicate implementation
repo. RSNA and Phoenix remain in their existing repositories until a measured
integration need justifies consolidation.

## D-002 Priority allocation

For the next 7 days: RSNA data unblock is P0, revenue validation runs in
parallel as P0/P1, Phoenix receives one controlled experiment, and agent
infrastructure work is limited to security and routing needed by those tasks.

## D-003 Data safety

Never commit DICOMs, raw medical datasets, checkpoints, secrets, or generated
submissions. Use versioned manifests, hashes, and external storage references.

## D-004 Experiment discipline

No experiment is valid without a hypothesis, baseline, single primary change,
authoritative metric, artifact path, interpretation, and next decision.

## D-005 Agent permissions

Default agent mode is least privilege: read-only research by default; writes,
code execution, external communication, credentials, destructive actions, and
submissions require explicit human approval or a narrowly scoped tool policy.

