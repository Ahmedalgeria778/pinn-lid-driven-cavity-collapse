# -*- coding: utf-8 -*-
"""Generate the Microsoft Word document with alt text for every figure and
table of the manuscript_3geometries manuscript (AIP Publishing accessibility
requirement). Nothing in manuscript.tex is modified.

Run:  py -3.11 results_reviewers/make_alt_text_docx_3geometries.py
Output: manuscript_3geometries/Alt_Text_for_Figures_and_Tables.docx
"""
import os

from docx import Document
from docx.shared import Pt

ROOT = os.path.dirname(os.path.abspath(__file__))          # results_reviewers
MS_DIR = os.path.join(os.path.dirname(ROOT), "manuscript_3geometries")

# (kind, label, caption-or-title, alt-text)
MAIN = [
    # ---- Figures -------------------------------------------------------
    ("Figure", "Figure 1",
     "Second-mode energy fraction $f_2$ (left) and time-dependent energy "
     "fraction $f_t$ (right) of the identified control versus Reynolds number "
     "for the Fourier and modified-Chebyshev parametrizations. For both "
     "parametrizations, $f_2$ and $f_t$ are computed by projection onto the "
     "common Fourier modal basis. All modal quantities are dimensionless and "
     "shown as percentages. The Fourier branch is strongly second-mode "
     "dominated at Re = 100 and 500, while the concentration weakens at "
     "Re = 1000.",
     "Two-panel line graph. Left panel: the second-mode energy fraction f2 "
     "(in percent) on the vertical axis versus Reynolds number on the "
     "horizontal axis, for the Fourier basis (solid curve with circles) and "
     "the modified-Chebyshev basis (dashed curve with squares). The Fourier "
     "curve is about 91.6 percent at Re = 100 and 92.2 percent at Re = 500, "
     "then drops to 64.0 percent at Re = 1000, while the Chebyshev curve stays "
     "near zero. Right panel: the time-dependent energy fraction ft (in "
     "percent) versus Reynolds number; the Fourier curve stays near or below "
     "1.3 percent at Re = 100 and 500 and rises to 32.2 percent at Re = 1000, "
     "while the Chebyshev curve exceeds 93 percent at all three Reynolds "
     "numbers."),
    ("Figure", "Figure 2",
     "Reynolds-number dependence of the LBM-realized response to the PINN "
     "Fourier control. Left: flow-response amplitude $a_2^{\\mathrm{flow}}$ "
     "extracted from the mean vertical-centerline velocity. Right: total "
     "enstrophy of the tested lid profiles. Coordinates, velocity, and "
     "enstrophy-related quantities are dimensionless.",
     "Two-panel graph of the independently realized flow response versus "
     "Reynolds number. Left panel: the flow-response amplitude a2 of the mean "
     "flow along the vertical centerline (dimensionless) on a logarithmic "
     "vertical axis versus Reynolds number (100, 500, 1000); the amplitude "
     "increases from about 3e-3 at Re = 100 to 5e-2 at Re = 1000. Right "
     "panel: total enstrophy (dimensionless) versus Reynolds number, lower "
     "for the controlled profiles than for the uniform lid at every Reynolds "
     "number."),
    ("Figure", "Figure 3",
     "Actuation-energy sweep at Re = 500. Left: second-mode energy fraction "
     "$f_2$. Right: coefficient $A_2$ of the stationary second spatial "
     "harmonic. $E^*$ and $E_r$ are dimensionless; modal fractions are shown "
     "as percentages.",
     "Two-panel graph with the target actuation energy E* from 0.01 to 2 on "
     "the horizontal axis (logarithmic scale). Left panel: second-mode energy "
     "fraction f2 (in percent) versus target energy; the fraction stays near "
     "92 percent for targets up to 0.25, then decreases to about 74 percent "
     "at E* = 2. Right panel: coefficient A2 of the stationary second spatial "
     "harmonic (dimensionless) versus target energy; A2 rises monotonically "
     "as the target energy increases."),
    ("Figure", "Figure 4",
     "Aspect-ratio dependence of the identified control branch at Re = 500 "
     "and $E^*=0.25$. (a) First three stationary spatial coefficients $A_1$, "
     "$A_2$, and $A_3$ of the identified Fourier control. (b) Fraction of "
     "control energy contained in the stationary second spatial harmonic, "
     "$f_2$, and in the explicitly time-dependent modes, $f_t$, for "
     "$AR=L_x/L_y=0.5$, $1$, and $2$. Coefficients and aspect ratios are "
     "dimensionless; modal fractions are reported as percentages.",
     "Two-panel grouped-bar chart comparing the three tested aspect ratios "
     "0.5, 1, and 2 (Lx/Ly). Panel (a): the stationary spatial Fourier "
     "coefficients A1, A2, and A3; A2 is the largest coefficient for every "
     "geometry (about 0.60 at AR = 0.5 and about 0.68 at AR = 1 and 2), "
     "while A1 and A3 stay small. Panel (b): the second-mode energy fraction "
     "f2 (about 70 percent at AR = 0.5, 92.2 percent at AR = 1, and 91.2 "
     "percent at AR = 2) and the time-dependent fraction ft (about 26.7 "
     "percent at AR = 0.5, 1.3 percent at AR = 1, and 1.6 percent at AR = 2)."),
    ("Figure", "Figure 5",
     "Identified lid controls $U_{\\mathrm{lid}}(s)$, with $s=x/L_x$, at "
     "Re = 500 and $E^*=0.25$ for $AR=L_x/L_y=0.5$ (a), $1$ (b), and $2$ (c). "
     "The complete PINN control is compared with its stationary second-"
     "harmonic contribution $A_2\\sin(2\\pi s)$ and the corresponding "
     "reduced-mode reconstruction. The increasing departure from the pure "
     "second-harmonic profile at $AR=0.5$ is consistent with the reduced "
     "value of $f_2$. Control velocity and normalized coordinate are "
     "dimensionless.",
     "Three-panel line graph of the identified lid velocity profile U_lid as "
     "a function of the normalized horizontal coordinate s = x/Lx, one panel "
     "per aspect ratio: (a) 0.5, (b) 1, and (c) 2. Each panel superposes the "
     "reference second-harmonic profile A2 sin(2 pi s), the complete PINN "
     "control, and its reduced-mode reconstruction. In panels (b) and (c) the "
     "three curves nearly coincide, following a positive lobe near the left "
     "wall and a negative lobe near the right wall; in panel (a) the complete "
     "control departs more visibly from the pure second harmonic."),
    ("Figure", "Figure 6",
     "Verification of the independent LBM-MRT solver against Ghia et al. for "
     "the uniform-lid cavity at Re = 100 and Re = 1000. The $u$-velocity "
     "profile along the vertical centerline ($x=0.5$) is compared with the "
     "reference data. Reynolds number and velocity are dimensionless.",
     "Two-panel line graph of the u-velocity profile along the vertical "
     "centerline (x = 0.5) as a function of the vertical coordinate y, for "
     "Re = 100 (left) and Re = 1000 (right). The LBM solution and the Ghia "
     "reference data are plotted on the same axes and agree to within a few "
     "percent, validating the uncontrolled solver."),
    ("Figure", "Figure 7",
     "Three-grid convergence study at Re = 500 for the stationary second-"
     "harmonic control. Left: integrated dissipation for $N=128$, $256$, and "
     "$512$. Right: deviation from the extrapolated value versus grid "
     "spacing, with observed order $p_{\\mathrm{obs}}=1.08$. Reynolds number, "
     "grid spacing, and dissipation are dimensionless.",
     "Two-panel graph of the grid-convergence study at grids of 128, 256, and "
     "512 points. Left panel: integrated viscous dissipation against grid "
     "size, increasing slightly from about 0.0094 to 0.0109 as the grid is "
     "refined toward the extrapolated value near 0.0114. Right panel: "
     "deviation of the dissipation from the extrapolated value versus grid "
     "spacing on logarithmic axes, with an observed convergence order p = "
     "1.08."),
    ("Figure", "Figure 8",
     "Integrated dissipation of the LBM-realized flows under the uniform lid, "
     "first spatial harmonic, stationary second harmonic, and complete PINN "
     "mean control. The percentages indicate the relative change with respect "
     "to the uniform-lid reference. All dissipation quantities are "
     "dimensionless.",
     "Grouped-bar chart with one bar group per Reynolds number (100, 500, "
     "1000). Each group shows the integrated dissipation of the uniform-lid, "
     "sin(pi x), A2 sin(2 pi x), and PINN mean controls, with the relative "
     "change with respect to the uniform reference shown as a percentage. The "
     "controlled cases are substantially lower than the uniform reference at "
     "every Reynolds number, with relative reductions between about 73 and 80 "
     "percent."),
    ("Figure", "Figure 9",
     "Consolidated identification and independent-verification results. (a) "
     "Integrated dissipation of the stationary second-harmonic control, "
     "complete PINN mean control, and uniform reference versus Reynolds "
     "number. (b) Boundary-control coefficient $a_2^{\\mathrm{ctrl}}$ and "
     "independently extracted flow-response amplitude "
     "$a_2^{\\mathrm{flow}}$. (c) Fluctuating-to-mean-flow kinetic ratio "
     "$R_K$ for the time-dependent Fourier and modified-Chebyshev branches, "
     "together with the adopted $1\\%$ threshold. (d) Dominant modal indices "
     "of the mean and fluctuation fields. All quantities are dimensionless "
     "except the displayed percentages.",
     "Four-panel summary graph. Panel (a): integrated dissipation versus "
     "Reynolds number for the stationary second-harmonic control, the "
     "complete PINN mean control, and the uniform reference, showing large "
     "reductions for the controlled cases. Panel (b): boundary-control "
     "coefficient a2 (control) and independently extracted flow-response "
     "amplitude a2 (flow) versus Reynolds number, showing a compatible trend "
     "with very different magnitudes. Panel (c): fluctuating-to-mean-flow "
     "kinetic-energy ratio RK (in percent) versus Reynolds number for the "
     "time-dependent Fourier and modified-Chebyshev branches, together with "
     "the adopted 1 percent threshold; the Fourier branch is below the "
     "threshold, the Chebyshev branch far above it. Panel (d): dominant "
     "modal indices of the mean and fluctuation fields for the two branches."),
    # ---- Tables --------------------------------------------------------
    ("Table", "Table I",
     "Baseline Fourier branch. $E_r$ is the realized control energy, $A_2$ "
     "the coefficient of the stationary second spatial harmonic, $f_2$ its "
     "energy fraction, and $f_t$ the time-dependent energy fraction. $A_2$ "
     "and $E_r$ are dimensionless; the fractions are percentages.",
     "Table with one row per Reynolds number (100, 500, 1000) and four "
     "columns: realized control energy Er, coefficient A2, second-mode "
     "fraction f2 (in percent), and time-dependent fraction ft (in percent). "
     "At Re = 100 and 500 the second-mode fraction is 91.6 and 92.2 percent "
     "with ft equal to 0.7 and 1.3 percent, while at Re = 1000 it drops to "
     "64.0 percent with ft = 32.2 percent."),
    ("Table", "Table II",
     "Actuation-energy sweep at Re = 500. $E^*$ is the prescribed "
     "dimensionless target, $E_r$ the realized dimensionless energy, $f_2$ "
     "the second-mode fraction, and $f_t$ the time-dependent fraction.",
     "Table of seven rows, one per target energy E* from 0.01 to 2.00, with "
     "columns E*, realized energy Er, second-mode fraction f2 (in percent), "
     "and time-dependent fraction ft (in percent). The second-mode fraction "
     "stays between 92.1 and 92.8 percent for targets up to 0.25 and "
     "decreases to 74.0 percent at E* = 2.00, while ft rises from 0.4 to 21.8 "
     "percent."),
    ("Table", "Table III",
     "Ten-seed training study at Re = 500. $E_r$ is realized control energy, "
     "$f_2$ the second-mode fraction, $f_t$ the time-dependent fraction, and "
     "$\\mathcal{L}_{\\mathrm{tot}}$ the final total loss. All quantities are "
     "dimensionless except the percentages.",
     "Table of ten rows, one per independent training seed (0 to 9), with "
     "columns Er, f2 (in percent), ft (in percent), and final total loss. The "
     "realized energies lie in 0.240-0.268 and the final losses in "
     "1.371-1.394, while the modal composition splits into four second-mode-"
     "dominated runs (seeds 1, 2, 6, 7), four temporally dominated runs "
     "(seeds 3, 5, 8, 9), and two intermediate runs (seeds 0, 4)."),
    ("Table", "Table IV",
     "Loss-ablation study at Re = 500. $E_r$ is realized control energy, "
     "$f_2$ the second-mode fraction, and $f_t$ the time-dependent fraction. "
     "All quantities are dimensionless except the percentages.",
     "Table with eight configuration rows and three columns (Er, f2 in "
     "percent, ft in percent). The baseline gives f2 = 92.2 percent; "
     "removing the variance regularization L_var redirects the solution to a "
     "first-harmonic branch with f2 near 0.2 percent (whether or not L_Re or "
     "L_diss is also removed), while halving or doubling the variance weight "
     "preserves the branch with f2 between 89.3 and 92.8 percent."),
    ("Table", "Table V",
     "Network-capacity and control-space truncation study at Re = 500. "
     "Architecture ranges are obtained from three seeds; basis cases are "
     "independent retrainings. $f_2$ is the second-mode fraction and $f_t$ "
     "the time-dependent fraction.",
     "Two-part table. Left: the three network architectures "
     "([64,64,32], [128,128,96,64], [256,256,128,96]) with the range of f2 "
     "(in percent) obtained over three seeds (90.3-93.6; 89.2-91.1 excluding "
     "the intermediate branch; 86.7-92.2). Right: control-space truncations "
     "(Nx,Nt) of (4,3), (6,1), (6,3), (6,5), (8,5), (8,9), corresponding to "
     "12 to 72 coefficients, with Er and f2 in percent; f2 ranges from 87.3 "
     "to 92.3 percent."),
    ("Table", "Table VI",
     "Aspect-ratio dependence of the identified control branch at Re = 500 "
     "and $E^*=0.25$. $A_2$ is the coefficient of the stationary second "
     "spatial harmonic, $E_r$ the realized control energy, $f_2$ the fraction "
     "of control energy contained in the stationary second harmonic, and "
     "$f_t$ the time-dependent control-energy fraction. All quantities are "
     "dimensionless except the percentages.",
     "Table of three rows (aspect ratios Lx/Ly = 0.5, 1, 2) with columns A2, "
     "Er, f2 (in percent), and ft (in percent). A2 is 0.6034 (AR = 0.5), "
     "0.6835 (AR = 1), and 0.6770 (AR = 2); Er is 0.2603, 0.2534, and 0.2514; "
     "f2 is 69.9, 92.2, and 91.2 percent; ft is 26.7, 1.3, and 1.6 percent."),
    ("Table", "Table VII",
     "Comparison of the Fourier and modified-Chebyshev control "
     "parametrizations. $E_r$ is the realized control energy, $f_2$ the "
     "projection onto the physical stationary second harmonic, and $f_t$ the "
     "time-dependent energy fraction. All energies and coefficients are "
     "dimensionless; fractions are percentages.",
     "Table of two block rows (Fourier and modified-Chebyshev bases), each "
     "with three Reynolds-number rows (100, 500, 1000), and columns Er, f2 "
     "(in percent), ft (in percent), and reconstruction RMSE. The Fourier "
     "basis gives f2 near 92 percent at Re = 100-500 and 64.0 percent at "
     "Re = 1000 with machine-precision reconstruction (RMSE of order 1e-16), "
     "whereas the modified-Chebyshev basis gives ft between 93 and 99 percent "
     "with reconstruction RMSE of order 1e-3."),
    ("Table", "Table VIII",
     "Centerline errors of the independent LBM-MRT solver relative to the "
     "Ghia benchmark. All error quantities are dimensionless.",
     "Table of two rows (Re = 100 and 1000) with the L2 and L-infinity errors "
     "of the u- and v-velocity components along the centerline. At Re = 100 "
     "the L2(u) error is 0.00506 and the L-infinity(u) error 0.01283; at "
     "Re = 1000 they are 0.01411 and 0.03003, with the v-component errors of "
     "similar magnitude."),
    ("Table", "Table IX",
     "Integrated dissipation of the steady LBM realizations. The uniform lid, "
     "first harmonic, stationary second harmonic, and time-mean PINN control "
     "are compared at three Reynolds numbers. Dissipation and Reynolds number "
     "are dimensionless.",
     "Table of three rows (Re = 100, 500, 1000) and four columns of "
     "integrated dissipation: uniform lid, sin(pi x), A2 sin(2 pi x), and "
     "PINN (mean). The stationary second-harmonic control gives the lowest "
     "dissipation at every Reynolds number (0.03445, 0.01041, 0.00473), the "
     "PINN mean control is close to it (0.03776, 0.01189, 0.00524), and the "
     "uniform-lid values (0.16195, 0.04425, 0.02654) are much larger."),
    ("Table", "Table X",
     "Time-dependent LBM assessment of the Fourier and modified-Chebyshev "
     "branches. $R_K=K_{\\mathrm{fluct}}/K_{\\mathrm{mean}}$ is dimensionless "
     "and reported as a percentage. The operational threshold is "
     "$R_K=1\\%$.",
     "Table of six rows (Fourier and modified-Chebyshev branches at Re = 100, "
     "500, 1000) with columns RK (in percent) and the pass/fail criterion. "
     "The Fourier branch passes the adopted 1 percent threshold at every "
     "Reynolds number (0.15, 0.04, 0.65 percent); the modified-Chebyshev "
     "branch fails by large margins (847, 44.0, 125 percent)."),
    ("Table", "Table XI",
     "Leave-one-condition-out evaluation of the symbolic representation on "
     "the auxiliary nine-condition dataset. $R^2$ is dimensionless.",
     "Table of one row of R2 values (coefficient of determination, "
     "dimensionless) over the nine held-out Reynolds conditions 100 to 1000; "
     "R2 ranges from 0.9326 to 0.9908, showing that a compact symbolic "
     "expression reproduces the identified control across the sampled "
     "conditions."),
]

TITLE = ("Physics-informed identification and independent numerical "
         "verification of conditional modal collapse in active control of "
         "lid-driven cavity flow")


def add_block(doc, kind, label, title, alt):
    h = doc.add_heading(f"{label} ({kind.lower()})", level=2)
    doc.add_paragraph().add_run("Caption/Title: ").bold = True
    doc.add_paragraph(title)
    doc.add_paragraph().add_run("Alt text: ").bold = True
    doc.add_paragraph(alt)


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)

    doc.add_heading("Alt text for figures and tables", level=0)
    doc.add_paragraph(
        "Physics of Fluids manuscript (manuscript_3geometries): "
        f"\u201c{TITLE}.\u201d "
        "This document provides the alt text for every figure and table of "
        "the main text, as requested by AIP Publishing for accessibility. "
        "It was generated automatically from manuscript.tex; no change was "
        "made to the manuscript itself."
    )

    doc.add_heading("Main text", level=1)
    for kind, label, title, alt in MAIN:
        add_block(doc, kind, label, title, alt)

    out = os.path.join(MS_DIR, "Alt_Text_for_Figures_and_Tables.docx")
    doc.save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()