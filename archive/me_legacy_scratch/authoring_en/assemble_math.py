#!/usr/bin/env python3
"""Assemble the authored concept fragments into a standalone English source.

Authoring helper for this request. The delivered full .tex is editable;
do not regenerate over subsequent direct edits without reconciling them.
This script never compiles or exports a PDF.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
parts = [HERE / name for name in ["math_foundations.tex", "math_roots_linear.tex", "math_roots_calculus.tex", "math_bridge.tex", "math_certification.tex"]]
entries = []
for path in parts:
    text = path.read_text()
    starts = list(re.finditer(r"\\concept\{([FBC]\d{2})\}\{([^\n]+)\}\{([^\n]+)\}", text))
    for i, match in enumerate(starts):
        body = text[match.end():starts[i+1].start() if i+1 < len(starts) else len(text)].strip()
        entries.append((match.group(1), match.group(2), match.group(3), body))
expected = [f"F{i:02}" for i in range(1,47)] + [f"B{i:02}" for i in range(1,31)] + [f"C{i:02}" for i in range(1,29)]
assert [e[0] for e in entries] == expected

preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Noto Serif}
\setsansfont{Noto Sans}
\setmonofont{Noto Sans Mono}[Scale=0.86]
\usepackage[english]{babel}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{geometry,booktabs,longtable,tabularx,array}
\usepackage{xcolor,hyperref,xurl,fancyhdr,enumitem}
\geometry{margin=2.2cm,headheight=15pt}
\definecolor{ink}{HTML}{183449}
\definecolor{teal}{HTML}{187C80}
\hypersetup{colorlinks=true,linkcolor=ink,citecolor=teal,urlcolor=teal,
pdftitle={ECDR Mathematics: Concepts, Foundations, and Proofs},pdfauthor={ECDR Research}}
\urlstyle{same}
\makeatletter
\renewcommand{\@pnumwidth}{2.5em}
\renewcommand{\@tocrmarg}{3.5em}
\renewcommand{\l@section}[2]{\@dottedtocline{1}{0em}{3em}{\bfseries#1}{#2}}
\makeatother
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\setlist{nosep,leftmargin=1.5em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\R}{\mathbb R}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\abs}[1]{\left\lvert#1\right\rvert}
\newcommand{\code}[1]{{\small\ttfamily\nolinkurl{#1}}}
\newcommand{\relu}{\operatorname{ReLU}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\E}{\mathbb E}
\newcommand{\conceptid}{Front matter}
\newcommand{\concept}[3]{\clearpage\renewcommand{\conceptid}{#1}%
\section{#2}\label{concept:#1}%
\textcolor{teal}{\textbf{Concept #1.} Prerequisites and connections: #3.}\par
\typeout{ECDR-CONCEPT-START #1 PAGE=\thepage}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small ECDR Mathematics | English Concept Atlas}
\fancyhead[R]{\small\conceptid}
\fancyfoot[L]{\small 28 October 2026}
\fancyfoot[R]{\thepage}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\vspace*{1cm}
{\Large\color{teal} FOUNDATIONS, DEFINITIONS, AND CONDITIONAL PROOFS\par}
\vspace{0.7cm}
{\huge\bfseries\color{ink} ECDR Mathematics:\\Concepts, Foundations,\\and Proofs\par}
\vspace{0.8cm}
{\Large One concept per page, from foundational axioms\\to certified decision reuse\par}
\vspace{1cm}
\begin{tabularx}{\linewidth}{@{}P{3.2cm}Y@{}}\toprule
Language & English throughout; previous Vietnamese documents remain available separately.\\
Reading unit & 104 concepts: 46 foundations, 30 data/model bridges, 28 certification and systems concepts.\\
Explanatory depth & Definition, every local symbol, derivation or example, motivation, assumptions and connections.\\
Meaning of span & Linear span, annotation temporal span, and receptive-field span are explained separately.\\
Reference model & Fixed preprocessing plus a CNN; valid/ReLU/GAP is a specified theoretical example, not a trained K12 result.\\
Evidence boundary & Mathematical derivations and declared examples; no new HAPT inference, formal executable verification or runtime benchmark.\\\bottomrule
\end{tabularx}
\vfill
This atlas expands the previous mathematical explanation. It traces the
assumptions actually used by ECDR back to the real-number, vector-space and
probability frameworks. It does not claim to prove axioms from nothing,
formalize all set theory, or establish a new foundation for mathematics.
\end{titlepage}
\hypersetup{pageanchor=true}\pagenumbering{arabic}
\section*{How to read a concept page}
\addcontentsline{toc}{section}{How to read a concept page}
Start with an unfamiliar term and follow its prerequisite IDs. Each concept
starts on a separate page. Front matter, coverage tables and references are
not concept pages. Symbols are explained locally so a page can be read
without reconstructing definitions from a distant chapter.

Four kinds of statements have different logical roles. An \textbf{axiom}
is accepted in the chosen mathematical framework. A \textbf{definition}
assigns meaning to notation. A \textbf{theorem} follows from stated
hypotheses. A \textbf{modeling assumption} chooses how a physical or
computational system is represented. A 50 Hz sampling model is not a
field axiom; a positive classifier margin is not an axiom of probability.

\textbf{The foundation chain.} Real-number algebra and order support scalar
inequalities. Vector-space structure supports linear combinations and
span. Inner products and norms support distance, projection and duality.
Induction, convexity, spectral algebra, calculus, optimization and
complexity are unpacked before their applications.
Continuity and compactness support existence statements. Probability
axioms support coverage statements. Sound operator bounds and a fixed
reference connect these tools to a universal decision property.

\textbf{Canonical pipeline notation.} A raw vector is $x\in\R^d$;
normalized input is $z=\mathcal P(x)$. The neural network is $f_\theta$,
the full logit map is $F=f_\theta\circ\mathcal P$, and the class is
$h(x)=\arg\max_jF_j(x)$ with a fixed tie rule. Raw pairwise margins are
$g_{cj}(x)=F_c(x)-F_j(x)$. In normalized coordinates,
$\gamma_{cj}(z)=f_{\theta,c}(z)-f_{\theta,j}(z)$, so
$g_{cj}=\gamma_{cj}\circ\mathcal P$. Tensor versions $X,Z$ retain
channel/time axes. Dimensions, index ranges and units are restated on the
relevant concept pages.

\textbf{Three uses of span.} F12 concerns the set of all linear combinations
of vectors. B02 concerns an annotated interval of time indices. B21
concerns the extreme input indices reachable by a feature. Shared English
vocabulary does not make these objects mathematically interchangeable.

\textbf{Limits.} Real-arithmetic proofs are conditional on their stated
reference and domain. Existing empirical artifacts are not new experiments.
No value of coverage, accuracy or speedup is inferred merely from a theorem.
See C20--C28 for executable soundness, lifecycle, cost and evidence boundaries.
\clearpage
\begingroup\small\setlength{\parskip}{0pt}\tableofcontents\endgroup
"""

def links(text):
    return re.sub(r"\b([FBC]\d{2})\b", lambda m:r"\hyperref[concept:"+m[1]+"]{"+m[1]+"}", text)

chunks = [preamble]
for cid,title,prereq,body in entries:
    chunks.append("\\concept{"+cid+"}{"+title+"}{"+links(prereq)+"}\n"+links(body)+
                  "\n\\par\\label{endconcept:"+cid+"}\n\\typeout{ECDR-CONCEPT-END "+cid+" PAGE=\\thepage}\n")
footer = (HERE / "math_appendices.tex").read_text()
chunks.append(footer)
(REPORT / "ECDR_Mathematics_Concepts_EN.tex").write_text("\n".join(chunks))
print(f"Assembled {len(entries)} concepts into standalone English source.")
