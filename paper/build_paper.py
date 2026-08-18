#!/usr/bin/env python3
"""Build the M6 two-column academic paper with ReportLab."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.52 * inch
GUTTER = 0.22 * inch
COLUMN_WIDTH = (PAGE_WIDTH - 2 * MARGIN - GUTTER) / 2
CONTENT_HEIGHT = PAGE_HEIGHT - 2 * MARGIN
TITLE_HEIGHT = 2.05 * inch

FIGURES = {
    "m1": ROOT / "results/m1_observables_20260818T002648-0400.png",
    "m2": ROOT / "results/m2_fixed_temperature_20260818T012603-0400.png",
    "m3": ROOT / "results/m3_conditioned_summary_20260818T015650-0400.png",
    "m4_summary": ROOT / "results/m4_tc_summary_20260818T021212-0400.png",
    "m4_trajectory": ROOT / "results/m4_trajectory_signatures_20260818T021212-0400.png",
}


def page_chrome(canvas, document) -> None:
    canvas.saveState()
    canvas.setTitle("Learning Criticality with a Temperature-Conditioned GFlowNet")
    canvas.setAuthor("Manik Sharma")
    canvas.setSubject("GFlowNet sampling and critical-temperature inference for the 2D Ising model")
    if document.page > 1:
        canvas.setStrokeColor(colors.HexColor("#777777"))
        canvas.setLineWidth(0.35)
        canvas.line(MARGIN, PAGE_HEIGHT - 0.34 * inch, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.34 * inch)
        canvas.setFont("Times-Italic", 7.3)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(MARGIN, PAGE_HEIGHT - 0.27 * inch, "Learning Criticality with a Temperature-Conditioned GFlowNet")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.27 * inch, "GFlowNet Ising Study")
    canvas.setFont("Times-Roman", 7.3)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.24 * inch, str(document.page))
    canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=sample["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=5,
            textColor=colors.HexColor("#17233a"),
        ),
        "author": ParagraphStyle(
            "Author",
            parent=sample["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "abstract_heading": ParagraphStyle(
            "AbstractHeading",
            parent=sample["Heading2"],
            fontName="Times-Bold",
            fontSize=9,
            leading=10,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=2,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=sample["Normal"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=9.6,
            alignment=TA_JUSTIFY,
        ),
        "body": ParagraphStyle(
            "PaperBody",
            parent=sample["BodyText"],
            fontName="Times-Roman",
            fontSize=8.35,
            leading=10.1,
            alignment=TA_JUSTIFY,
            spaceAfter=4.2,
            splitLongWords=True,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=sample["Heading1"],
            fontName="Times-Bold",
            fontSize=11.0,
            leading=12.4,
            alignment=TA_LEFT,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
            textColor=colors.HexColor("#17233a"),
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=sample["Heading2"],
            fontName="Times-BoldItalic",
            fontSize=9.2,
            leading=10.4,
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=sample["Normal"],
            fontName="Times-Italic",
            fontSize=8.5,
            leading=10,
            alignment=TA_CENTER,
            leftIndent=5,
            rightIndent=5,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["Normal"],
            fontName="Times-Roman",
            fontSize=7.4,
            leading=8.7,
            alignment=TA_JUSTIFY,
            spaceBefore=4,
            spaceAfter=3,
        ),
        "source": ParagraphStyle(
            "Source",
            parent=sample["Normal"],
            fontName="Courier",
            fontSize=5.8,
            leading=7,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#555555"),
            spaceAfter=6,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=sample["Normal"],
            fontName="Times-Roman",
            fontSize=7.7,
            leading=9.2,
            alignment=TA_LEFT,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=2,
        ),
    }


def academic_table(rows, widths, font_size=6.9) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17233a")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#8a94a3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def scaled_image(path: Path, maximum_width: float, maximum_height: float) -> Image:
    pixel_width, pixel_height = ImageReader(str(path)).getSize()
    scale = min(maximum_width / pixel_width, maximum_height / pixel_height)
    return Image(str(path), width=pixel_width * scale, height=pixel_height * scale)


def wide_figure_page(
    story: list,
    styles: dict[str, ParagraphStyle],
    figure_key: str,
    number: str,
    caption: str,
    max_height: float = 6.1 * inch,
    return_to_two_column: bool = True,
) -> None:
    path = FIGURES[figure_key]
    flowables = [
            NextPageTemplate("Wide"),
            PageBreak(),
            Spacer(1, 0.10 * inch),
            scaled_image(path, PAGE_WIDTH - 2 * MARGIN, max_height),
            Paragraph(f"<b>Figure {number}.</b> {caption}", styles["caption"]),
            Paragraph(f"Artifact: {path.relative_to(ROOT)}", styles["source"]),
    ]
    if return_to_two_column:
        flowables.extend((NextPageTemplate("TwoColumn"), PageBreak()))
    story.extend(flowables)


def add_body(story: list, styles: dict[str, ParagraphStyle]) -> None:
    p = lambda text: story.append(Paragraph(text, styles["body"]))
    heading = lambda text: story.append(Paragraph(text, styles["section"]))
    subheading = lambda text: story.append(Paragraph(text, styles["subsection"]))
    equation = lambda text: story.append(Paragraph(text, styles["equation"]))

    heading("1. Introduction")
    p(
        "The two-dimensional zero-field Ising model is a compact test of generative sampling near a phase transition. "
        "Below criticality it has two symmetry-related ordered modes; around criticality it develops long correlations; "
        "and its partition function is normally the hard normalization in the Boltzmann law. This study asks whether a "
        "small generative flow network (GFlowNet), trained on one Mac CPU, can reproduce the distribution and predict "
        "critical temperature from its learned structure rather than merely copying a known answer."
    )
    p(
        "The study has three safeguards against impressive-looking but incorrect results. First, all 65,536 states of "
        "the L=4 lattice are exactly enumerated. Second, seeded Metropolis-Hastings sampling provides an independent "
        "baseline at every tested temperature. Third, milestone validators enforce numerical thresholds and write "
        "immutable JSON provenance. The exact thermodynamic-limit value, T<sub>c</sub>=2/ln(1+sqrt(2))=2.269185, is "
        "reserved for reference and is never supplied to training [4]."
    )
    p(
        "The central contribution is two independent routes. Calibrating the learned-log-Z route by its 6.5% L=4 "
        "known-truth miss gives T<sub>c</sub>(log Z)=2.34 +/- 0.15. The observable route is kept as two estimates: "
        "susceptibility extrapolation 2.274 and Binder mean 2.246, a range of about 2.25 to 2.27. Their finite-size "
        "biases have opposite signs, so averaging them overstates precision."
    )
    p(
        "The computation and original report build ran in one automated session of roughly 90 minutes wall-clock."
    )

    heading("2. Model and Methods")
    subheading("2.1 Ising target and observables")
    p(
        "Spins s<sub>i</sub> are -1 or +1 on an L by L periodic square lattice with nearest-neighbor coupling J=1 and "
        "zero field. Counting every bond once, the Hamiltonian and Boltzmann law are"
    )
    equation("E(x) = - sum&lt;ij&gt; s<sub>i</sub>s<sub>j</sub>, &nbsp;&nbsp; p<sub>T</sub>(x) = exp[-E(x)/T] / Z(T).")
    p(
        "With N=L<super>2</super> and M=sum<sub>i</sub>s<sub>i</sub>, the analysis records energy per site, absolute "
        "magnetization, connected absolute-magnetization susceptibility, specific heat, and Binder cumulant:"
    )
    equation("chi = beta [ &lt;M<super>2</super>&gt; - &lt;|M|&gt;<super>2</super> ] / N, &nbsp;&nbsp; U<sub>4</sub> = 1 - &lt;M<super>4</super>&gt; / (3&lt;M<super>2</super>&gt;<super>2</super>).")
    p(
        "The absolute-magnetization convention prevents finite-volume tunneling between the two ordered signs from "
        "appearing as a spurious susceptibility. Exact L=4 values use stable log-sum-exp evaluation. Metropolis uses "
        "checkerboard single-spin sweeps with acceptance min[1, exp(-Delta E/T)], mixed starts, burn-in, thinning, and "
        "explicit independent seeds."
    )

    subheading("2.2 Raster-order GFlowNet")
    p(
        "The source is an unassigned lattice. At raster step t, the forward policy assigns -1 or +1 to site t. The "
        "terminal reward is R<sub>T</sub>(x)=exp[-E(x)/T]. Raster order creates one trajectory per terminal and one parent "
        "per non-source state, so backward transition probabilities equal one. General trajectory balance [2] reduces to"
    )
    equation("delta(x,T) = log Z(T) + sum<sub>t</sub> log P<sub>F</sub>(s<sub>t+1</sub>|s<sub>t</sub>,T) + E(x)/T.")
    p(
        "Training minimizes the mean squared residual. When balance holds for every complete trajectory, policy "
        "normalization forces P<sub>F</sub>(x|T)=R<sub>T</sub>(x)/Z(T). Thus log Z is learned rather than supplied by the "
        "oracle. This is the normalized discrete-model use of GFlowNets described in Refs. [1-3]."
    )

    subheading("2.3 Temperature conditioning and symmetry")
    p(
        "Fixed-temperature L=4 models use two 128-unit hidden layers. The conditioned models take beta=1/T and use a "
        "small separate network for log Z(beta). A masked autoregressive MLP returns every site conditional in one "
        "training pass while blocking present and future spins. The L=8 policy uses two 256-unit hidden layers, the CPU "
        "ceiling, plus spin and beta-times-spin features."
    )
    p(
        "Zero-field symmetry is imposed exactly by replacing a raw logit g(s,beta) with [g(s,beta)-g(-s,beta)]/2. "
        "Consequently q(x|beta)=q(-x|beta), the first spin is unbiased, and neither ordered sign is structurally favored. "
        "Training batches combine uniform, current-policy, noisy ordered, and block-domain configurations, paired with "
        "global flips. This off-policy mixture preserves full support while exposing low-energy and domain-wall sectors."
    )

    heading("3. Validation Results")
    subheading("3.1 Physics core and fixed-temperature models")
    p(
        "The final test suite contains 20 passing checks covering energy signs, wrapped bonds, exact enumeration, seeded "
        "sampling, autoregressive normalization and causality, spin symmetry, and numerical peak estimators. Table 1 "
        "shows the worst L=4 Metropolis deviations over the M1 temperature grid."
    )
    story.append(
        academic_table(
            [
                ["Metric", "Worst error", "Gate"],
                ["Energy/site", "0.1003%", "1%"],
                ["|m|", "0.1019%", "1%"],
                ["Susceptibility", "2.7492%", "5%"],
                ["Specific heat", "1.2070%", "5%"],
            ],
            [0.98 * inch, 0.72 * inch, 0.48 * inch],
        )
    )
    story.append(Paragraph("<b>Table 1.</b> Exact-oracle validation. Source: M1 JSON.", styles["caption"]))
    p(
        "For M2, two million seeded multinomial draws from exactly enumerated model probabilities give "
        "KL(exact||empirical)=0.014762 at T=3 and 0.015377 at T=2. For this normalized autoregressive model the draws "
        "are distributionally equivalent to sequential rollout terminals, but they were not rollouts. Exact enumerated "
        "model KL values are 0.000511 and 0.002200, and learned log Z errors are 0.0055% and 0.5473%. The histogram KL "
        "uses a disclosed Jeffreys half-count so physically nonzero rare states do not receive zero probability."
    )

    subheading("3.2 One conditioned model")
    p(
        "The single conditioned L=4 model reaches exact KL 0.003255, 0.001690, and 0.000308 at T=1.8, 2.269185, and 3.0. "
        "Its maximum log Z value error over a 16-point grid is 0.1053%. Table 2 compares 200,000 L=8 states from true "
        "sequential rollouts with 256,000 fresh Metropolis states at each temperature. M4 likewise used true sequential "
        "rollouts for its generated-observable curves."
    )
    story.append(
        academic_table(
            [
                ["T", "E error", "|m| error", "chi error"],
                ["1.8", "0.0034%", "0.0281%", "4.3499%"],
                ["2.269", "0.2359%", "0.4480%", "2.7736%"],
                ["3.0", "0.2462%", "0.3822%", "0.6978%"],
            ],
            [0.38 * inch, 0.56 * inch, 0.62 * inch, 0.62 * inch],
            font_size=6.6,
        )
    )
    story.append(Paragraph("<b>Table 2.</b> Conditioned L=8 relative errors. Source: M3 JSON.", styles["caption"]))
    p(
        "At T=1.8 the generated fraction P(m&gt;0)=0.498825, meeting the 0.50 +/- 0.05 mode criterion. At higher T the raw "
        "positive fraction falls slightly because exactly zero magnetization becomes more common; conditional on nonzero "
        "magnetization the signs remain balanced."
    )

    wide_figure_page(
        story,
        styles,
        "m1",
        "1",
        "Exact L=4 observables and seeded Metropolis baselines at L=4, 8, and 12. Every temperature-axis panel marks the exact thermodynamic-limit critical temperature. Source: M1.",
        return_to_two_column=False,
    )
    wide_figure_page(
        story,
        styles,
        "m2",
        "2",
        "Fixed-temperature trajectory-balance convergence and energy-level probability mass. Empirical model mass overlays exact enumeration at both temperatures. Source: M2.",
    )

    heading("4. Predicting Critical Temperature")
    subheading("4.1 Learned partition-function route")
    p(
        "The partition function supplies thermodynamics before terminal sampling. In beta coordinates, U=-d log Z/d beta "
        "and c=beta<super>2</super> d<super>2</super>log Z/d beta<super>2</super>/N. M4 evaluates log Z on 256 beta points, "
        "fits a degree-10 Chebyshev polynomial, differentiates it analytically, and searches for the interior heat-capacity "
        "maximum."
    )
    p(
        "Direct exact L=4 observables peak at T=2.438950, while the exact-log-Z pipeline returns 2.439257, a 0.0126% "
        "location error. This validates differentiation on exact data, but not the learned log Z itself. The learned "
        "L=4 peak is 2.281360, a 6.5% known-truth miss. Applying that empirical relative calibration error to the raw "
        "L=8 central peak gives"
    )
    equation("T<sub>c</sub><super>(log Z)</super> = 2.34 +/- 0.15.")
    p(
        "This is an empirical calibration bar, not a statistical confidence interval. The differentiation pipeline was "
        "validated on exact data, but the learned log Z itself carries this calibration error. The central value is in "
        "the predeclared [2.1,2.5] window. The differentiated learned curves also become negative near T=1.5; a canonical "
        "Ising heat capacity cannot be negative, so this is nonphysical boundary curvature, not a physical discovery."
    )

    subheading("4.2 Generated-observable route")
    p(
        "One hundred thousand new GFlowNet samples at each of 20 temperatures supply L=4 and L=8 observables; 144,000 new "
        "Metropolis samples per temperature supply L=12. Local quadratic fits place susceptibility maxima at 2.812028, "
        "2.552567, and 2.447225 for L=4, 8, and 12. Fitting T<sub>chi</sub>(L)=T<sub>c</sub>+a/L gives"
    )
    equation("T<sub>c</sub><super>(chi)</super> = 2.273526, &nbsp;&nbsp; R<super>2</super> = 0.99825.")
    p(
        "As a methodological control, the identical procedure on the existing M1 Metropolis-only L=4, 8, and 12 curves "
        "gives peaks 2.830413, 2.550983, and 2.446206, followed by"
    )
    equation("T<sub>c</sub><super>(chi, M1 MCMC)</super> = 2.259472, &nbsp;&nbsp; R<super>2</super> = 0.99941.")
    p(
        "No new sampling was performed. The coarser M1 grid makes this a method control rather than a precision result: "
        "switching from a five-point to a three-point local fit shifts its intercept by about 0.008. Linear Binder "
        "crossings are 2.242534 and 2.249908, with mean 2.246221. Thus the observable estimates are susceptibility 2.274 "
        "and Binder 2.246, a range of about 2.25 to 2.27. The first is 0.19% high and the second 1.01% low; their biases "
        "have opposite signs, so averaging them overstates precision. No observable consensus is reported."
    )

    subheading("4.3 Trajectory signatures")
    p(
        "The L=8 mean Bernoulli action entropy rises from 0.04587 nats at T=1.5 through 0.28136 at exact criticality to "
        "0.56129 at T=3.2. This smooth increase reflects the change from ordered, predictable later actions to disordered "
        "choices. At exact criticality P(m&gt;0|m!=0)=0.50014. These are exploratory structural signatures; no third critical "
        "temperature is extracted from them."
    )

    wide_figure_page(
        story,
        styles,
        "m3",
        "3",
        "Validation of one temperature-conditioned policy per lattice size: exact L=4 KL and log Z, L=8 observables against Metropolis, and terminal-mode balance. Source: M3.",
        return_to_two_column=False,
    )

    # Both wide M4 figures fit on one page.
    story.extend([NextPageTemplate("Wide"), PageBreak(), Spacer(1, 0.05 * inch)])
    story.append(scaled_image(FIGURES["m4_summary"], PAGE_WIDTH - 2 * MARGIN, 2.55 * inch))
    story.append(
        Paragraph(
            "<b>Figure 4.</b> The required M4 summary: heat capacity from learned log Z, susceptibility, and Binder cumulant, all with the exact critical-temperature line. Source: M4.",
            styles["caption"],
        )
    )
    story.append(Paragraph(f"Artifact: {FIGURES['m4_summary'].relative_to(ROOT)}", styles["source"]))
    story.append(scaled_image(FIGURES["m4_trajectory"], PAGE_WIDTH - 2 * MARGIN, 2.55 * inch))
    story.append(
        Paragraph(
            "<b>Figure 5.</b> Exploratory L=8 policy entropy by raster step and temperature, mean entropy, and positive-mode balance. Source: M4.",
            styles["caption"],
        )
    )
    story.append(Paragraph(f"Artifact: {FIGURES['m4_trajectory'].relative_to(ROOT)}", styles["source"]))
    story.extend([NextPageTemplate("TwoColumn"), PageBreak()])

    heading("5. Discussion")
    p(
        "The strongest evidence is the agreement of several independent objects: exact L=4 distributions, learned log Z "
        "values, L=8 generated observables, L=12 Metropolis trends, and separate finite-size critical estimators. The GFlowNet "
        "does more than match a mean. Its normalized autoregressive distribution covers both ordered signs and its learned "
        "normalization contains enough curvature to locate an interior heat-capacity peak."
    )
    p(
        "Finite size remains the dominant physical limitation. L=4 and L=8 are far from the thermodynamic limit; their "
        "rounded susceptibility peaks shift substantially. The high R-squared of the three-point 1/L line only describes "
        "those three points and is not asymptotic evidence. Repeated training seeds and bootstrap confidence intervals are "
        "also absent, so displayed digits identify this reproducible run rather than universal precision."
    )
    p(
        "Derivative sensitivity is the principal model limitation. Sub-percent errors in log Z values can become visible "
        "curvature errors after two derivatives, which is why the L=4 known-truth miss calibrates the L=8 result. A future "
        "model should parameterize log Z as a convex function of beta, "
        "because its second derivative is Var(E) and must be nonnegative. Larger locality-aware policies and repeated-seed "
        "uncertainty estimates are natural follow-ups."
    )
    p(
        "MCMC still wins for a one-off trusted estimate on a modest lattice: it needs no neural training, its transition "
        "rule is transparent, and it reached L=12 under the same CPU budget. The GFlowNet wins after amortization when many "
        "independent samples across temperature are required, when explicit two-mode generation matters, or when learned "
        "log Z is itself the object of study. The methods are complementary rather than interchangeable."
    )

    heading("6. Conclusion")
    p(
        "A compact CPU GFlowNet learned the periodic 2D Ising distribution to exact-oracle accuracy at L=4, generalized "
        "across temperature, and reproduced L=8 Metropolis observables within the declared gates. The calibrated learned "
        "partition function gives T<sub>c</sub>=2.34 +/- 0.15. Generated observables give separate susceptibility and "
        "Binder estimates of 2.274 and 2.246, spanning about 2.25 to 2.27 around exact 2.2692. Their opposite-sign biases "
        "are not averaged into a falsely precise consensus. The log-Z route remains the distinctively model-based result "
        "and reveals the numerical fragility of thermodynamic derivatives."
    )

    heading("Artifact Provenance")
    p(
        "Every quantitative claim above is regenerated by an acceptance-gated validator. Checkpoints are SHA-256 hashed in "
        "their JSON records. The report and this PDF introduce no uncomputed table entries."
    )
    story.append(
        academic_table(
            [
                ["ID", "Role", "Primary artifact"],
                ["M1", "Exact + MCMC", "results/m1_metrics_20260818T002648-0400.json"],
                ["M2", "Fixed-T GFN", "results/m2_metrics_20260818T012603-0400.json"],
                ["M3", "Conditioned GFN", "results/m3_metrics_20260818T015650-0400.json"],
                ["M4", "Criticality", "results/m4_metrics_20260818T021212-0400.json"],
                ["M5", "Report audit", "results/m5_metrics_20260818T030253-0400.json"],
            ],
            [0.25 * inch, 0.62 * inch, 1.80 * inch],
            font_size=5.5,
        )
    )
    story.append(Paragraph("<b>Table 3.</b> Machine-readable provenance map.", styles["caption"]))

    heading("References")
    references = [
        "[1] Y. Bengio et al., “GFlowNet Foundations,” arXiv:2111.09266 (2021).",
        "[2] N. Malkin et al., “Trajectory Balance: Improved Credit Assignment in GFlowNets,” arXiv:2201.13259 (2022).",
        "[3] D. Zhang et al., “Generative Flow Networks for Discrete Probabilistic Modeling,” arXiv:2202.01361 (2022).",
        "[4] L. Onsager, “Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition,” Physical Review 65, 117-149 (1944).",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["reference"]))


def build(output_path: Path) -> None:
    for path in FIGURES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()

    title_frame = Frame(
        MARGIN,
        PAGE_HEIGHT - MARGIN - TITLE_HEIGHT,
        PAGE_WIDTH - 2 * MARGIN,
        TITLE_HEIGHT,
        leftPadding=3,
        rightPadding=3,
        topPadding=3,
        bottomPadding=3,
        id="title",
    )
    lower_height = CONTENT_HEIGHT - TITLE_HEIGHT - 0.08 * inch
    first_left = Frame(
        MARGIN,
        MARGIN,
        COLUMN_WIDTH,
        lower_height,
        leftPadding=2,
        rightPadding=5,
        topPadding=3,
        bottomPadding=3,
        id="first_left",
    )
    first_right = Frame(
        MARGIN + COLUMN_WIDTH + GUTTER,
        MARGIN,
        COLUMN_WIDTH,
        lower_height,
        leftPadding=5,
        rightPadding=2,
        topPadding=3,
        bottomPadding=3,
        id="first_right",
    )
    later_left = Frame(
        MARGIN,
        MARGIN,
        COLUMN_WIDTH,
        CONTENT_HEIGHT,
        leftPadding=2,
        rightPadding=5,
        topPadding=12,
        bottomPadding=5,
        id="left",
    )
    later_right = Frame(
        MARGIN + COLUMN_WIDTH + GUTTER,
        MARGIN,
        COLUMN_WIDTH,
        CONTENT_HEIGHT,
        leftPadding=5,
        rightPadding=2,
        topPadding=12,
        bottomPadding=5,
        id="right",
    )
    wide_frame = Frame(
        MARGIN,
        MARGIN,
        PAGE_WIDTH - 2 * MARGIN,
        CONTENT_HEIGHT,
        leftPadding=3,
        rightPadding=3,
        topPadding=12,
        bottomPadding=5,
        id="wide",
    )
    document = BaseDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Learning Criticality with a Temperature-Conditioned GFlowNet",
        author="Manik Sharma",
        subject="2D Ising GFlowNet sampling and critical-temperature inference",
    )
    document.addPageTemplates(
        [
            PageTemplate("First", [title_frame, first_left, first_right], onPage=page_chrome),
            PageTemplate("TwoColumn", [later_left, later_right], onPage=page_chrome),
            PageTemplate("Wide", [wide_frame], onPage=page_chrome),
        ]
    )

    story: list = [
        Paragraph("Learning Criticality with a Temperature-Conditioned GFlowNet", styles["title"]),
        Paragraph(
            "Manik Sharma  |  University of Massachusetts Boston  |  August 2026",
            styles["author"],
        ),
        HRFlowable(width="100%", thickness=0.55, color=colors.HexColor("#66758c"), spaceBefore=1, spaceAfter=3),
        Paragraph("Abstract", styles["abstract_heading"]),
        Paragraph(
            "A temperature-conditioned GFlowNet is trained to sample the periodic two-dimensional Ising model and expose criticality through its learned normalization. Exact L=4 enumeration and seeded Metropolis chains through L=12 gate every result. Fixed-temperature empirical KL is below 0.016 nats; a conditioned model reaches L=4 exact KL below 0.0033 and L=8 observable errors below 4.35%. Calibrating the learned-log-Z route by its 6.5% L=4 known-truth miss gives T<sub>c</sub>=2.34 +/- 0.15. Susceptibility extrapolation gives 2.274 and the Binder mean 2.246, an observable range of about 2.25 to 2.27 around exact 2.2692; their opposite-sign biases are not averaged. The study identifies nonphysical boundary curvature, finite-size bias, and regimes where MCMC remains preferable.",
            styles["abstract"],
        ),
        Spacer(1, 3),
        Paragraph(
            "<b>Keywords:</b> GFlowNet; Ising model; trajectory balance; partition function; critical temperature; finite-size scaling",
            styles["abstract"],
        ),
        FrameBreak(),
        NextPageTemplate("TwoColumn"),
    ]
    add_body(story, styles)
    document.build(story)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/ising_gflownet_paper.pdf",
        help="PDF output path",
    )
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    build(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
