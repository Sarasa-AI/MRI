# MRI Portfolio Control Plane

This repository is the control plane for the Sarasa AI portfolio. It does not
contain the RSNA or Phoenix implementations. The canonical implementation
repositories currently live at:

- RSNA research: `/Users/sarasa/Documents/SARASA_kaggle`
- Phoenix clinical DR: `/Users/sarasa/Documents/phonix`
- MRI control plane: `/Users/sarasa/Documents/MRI/MRI`

Always run Git commands from the nested canonical path above. The parent
directory `/Users/sarasa/Documents/MRI` is a workspace wrapper, not a Git repo.

This repo stores verified state, decisions, priorities, risks, and execution
plans. Large datasets, DICOM files, checkpoints, credentials, and generated
outputs stay outside Git and must be represented by manifests and fingerprints.

Start with [PROJECT_STATE.md](PROJECT_STATE.md), then [ROADMAP.md](ROADMAP.md)
and [DECISIONS.md](DECISIONS.md).
