# hei-adjunct-paradox
Open Science Policy Flight Simulator, System Dynamics Verification Suite (8,192 Sobol runs, Barlas, Richardson, LTM), and Synthetic Scenarios for "The Adjunct Absorption Paradox in Engineering Higher Education" (Case HEI-A).
# The Adjunct Absorption Paradox in Engineering Higher Education (Case HEI-A)


[![Live Simulator](https://img.shields.io/badge/Live_SD_Simulator-GitHub_Pages-008080?style=for-the-badge)](https://japarravalencia.github.io/hei-adjunct-paradox/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)


Open Science replication package and browser-native interactive policy flight simulator accompanying:


> **Parra Valencia, J. A. (2026).** *The Adjunct Absorption Paradox in Engineering Higher Education: A System Dynamics Policy Analysis of Faculty Workload, Retention, and Accreditation Sustainability (2024–2030).*


## Quick Links
- **Interactive Web Flight Simulator (GitHub Pages):** https://japarravalencia.github.io/hei-adjunct-paradox/
- **Source Repository (GitHub):** https://github.com/japarravalencia/hei-adjunct-paradox


## Contents (Anonymized Case HEI-A, Zero PII)
- `index.html` — Browser-native continuous-time System Dynamics simulator (2024–2030, `dt = 0.25` yr) implementing all 23 constants and 6 verbatim piecewise-linear lookup curves (`g1–g6`), toggles for PMA retention support and outreach, full-time attrition stress slider (`0–8%/yr` exposing the `0.40` CNA floor breach), adjunct equivalence (`e_HC`), and CSV export.
- `undergrad_eng_sd_verification.py` — Deterministic Python verification suite reproducing all 6 baseline/policy/extreme scenarios, quantitative LTM dominance (`2024–2030`), multivariable Richardson convergence (`dt = 0.25 / 0.125 / 0.0625` yr + RK4), 8 Forrester–Senge/Barlas validation tests, dedicated structural stress sweeps (`alpha_pma`, `e_HC`, `r_ret,TC`), and Saltelli–Sobol global sensitivity (`n = 1,024`, `N = 8,192` runs, 95% bootstrap CIs).
- `sobol_ides_unicamacho.csv` & `corridas_csv/` — Precomputed verified synthetic trajectories and sensitivity indices.


## Institutional Confidentiality Statement
To protect institutional privacy regarding internal payroll cost ratios, tuition structures, and accreditation self-assessment records, the modeled engineering program is anonymized as **Case HEI-A**. Zero individual student or faculty records are included.
```

