# Exhaustive scan, N = 6 to 9

`edrn_orbit_n8_n9.py` scans every connected graph on N nodes for edge pairs from different automorphism orbits whose defect-response curves are equal. The result file holds the totals for N = 6, 7, 8, and 9.

Run (N = 9 takes several hours on 4 cores):

    python -X utf8 edrn_orbit_n8_n9.py --n 6 7 8
    python -X utf8 edrn_orbit_n8_n9.py --n 9 --workers 4

Requirements: python-igraph, numpy.

| N | connected graphs | different-orbit pairs, equal curves | graphs containing one | same-orbit pairs, different curves |
|---|---|---|---|---|
| 6 | 112 | 32 | 11 | 0 |
| 7 | 853 | 382 | 98 | 0 |
| 8 | 11,117 | 3,452 | 981 | 0 |
| 9 | 261,080 | 35,930 | 13,042 | 4 |

Notes:

- The Hamiltonian is built only in the sector S_z = 0 (even N) or -1/2 (odd N). The scalar sigma.sigma has the same expectation as in the full space. The script fails if N = 7 does not give 382 with no same-orbit counterexample.
- The N = 6 value 32 is one definition. The 47 in the abstract is a union over 24 analysis choices.
- The 4 same-orbit pairs at N = 9 sit in two graphs with a nearly degenerate ground state. Their curves differ by about the tolerance (1e-9), so they read as numerical.
- Among the 135,156 graphs without a non-trivial automorphism (8, 144, 3,552, and 131,452 at N = 6 to 9) no pair has equal curves. This holds up to N = 9. There is no proof.
- The counts depend on the 1e-9 tolerance. Borderline near-degenerate graphs could shift the totals slightly.
