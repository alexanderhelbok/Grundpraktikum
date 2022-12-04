\author{Alexander Helbok}
\date{\today}
\ProvidesClass{alex_bericht}[2022/10/12 Alexander Report Class]
\LoadClass[11pt,a4paper]{scrreprt}
\RequirePackage{fullpage, lastpage}
\RequirePackage[onehalfspacing]{setspace} 
\RequirePackage[margin=2.5cm]{geometry}
%\RequirePackage[utf8]{inputenc} % uncomment if xelatex is not used
\RequirePackage[T1]{fontenc}
\RequirePackage{fontspec}
\setmainfont[Mapping=tex-text]{Times New Roman}
\RequirePackage[backend=biber, abbreviate=true,doi=false,style=numeric-comp,giveninits=true,sorting=none]{biblatex}
\RequirePackage[ngerman]{babel}
\RequirePackage{xifthen, xstring, csquotes}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{mtpro2, lipsum}
\pdfmapfile{=mtpro2.map}
\RequirePackage{mathtools, amsthm, amsfonts, siunitx, physics, chemformula, empheq}
\RequirePackage{enumitem, multicol, tcolorbox, array, booktabs, vwcol}
\tcbuselibrary{breakable,skins}
\RequirePackage[headsepline]{scrlayer-scrpage}
\RequirePackage{graphicx, wrapfig, float}
\RequirePackage[font=footnotesize, bf, format=plain, font=normalsize]{caption} 
\RequirePackage{listings} %env ignores LATEX commands, keeping all the line breaks (for printing code)
\RequirePackage[
	pdftitle={Bericht 1},
	pdfsubject={},
	pdfauthor={Alexander Helbok},
	pdfkeywords={},	
	%Links nicht einrahmen
	hidelinks
]{hyperref}
\RequirePackage{bookmark, makecell}
\RequirePackage{custom}
\RequirePackage{tikz, pgfplots}
\usetikzlibrary{arrows.meta,positioning, calc, shapes, math, decorations.pathreplacing, decorations.markings}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{
	compat=newest,
	legend image code/.code={
		\draw[mark repeat=2,mark phase=2]
		plot coordinates {
			(0cm,0cm)
			(0.25cm,0cm)        %% default is (0.3cm,0cm)
			(0.4cm,0cm)         %% default is (0.6cm,0cm)
		};%
	}
}

\RedeclareSectionCommand[
beforeskip=-3sp,
afterskip=1\baselineskip]{chapter}
\RedeclareSectionCommand[
beforeskip=-\baselineskip,
afterskip=.5\baselineskip]{section}

%\tikzexternalize[prefix=figures/]

\graphicspath{ {./Graphics/} }
\addbibresource{Literatur.bib}

\sisetup{per-mode = symbol, sticky-per}%
\DeclareSIUnit[]\v{\m\per\s}
\DeclareSIUnit[]\a{\m\per\square\s}

% define häufig used cmd
\renewcommand*\dd{\mathop{}\!\mathrm{d}}
\renewcommand*{\vector}[1]{\hspace{0.1cm} \begin{pmatrix}#1\end{pmatrix}}
\newcommand*{\iu}{{i\mkern1mu}}
\newcommand*{\dl}[1]{\underline{\underline{#1}}}	%dl = double underline
\newcommand\supfrac[3][]{\mkern-2mu#1\frac{#2}{#3}\rule[-6pt]{0pt}{0pt}}

\newcolumntype{L}{>{$}c<{$}} % math-mode version of "l" column type

% expo function \expo[sign][nominator/denumerator][normal] (all optional)
\ExplSyntaxOn
\NewDocumentCommand{\expo}{>{\SplitArgument{1}{/}} O{} >{\SplitArgument{1}{/}} O{} O{} }{%
	\ifthenelse{\equal{\use_i:nn#1}{-}}
	{\regex_match:nnTF {NoValue}{\use_ii:n#2}
		{\,\mathrm{e}^{\mkern-2mu\use_i:nn#1\rule[-6pt]{0pt}{0pt}\use_i:nn#2}}
		{\,\mathrm{e}^{\mkern-2mu\use_i:nn#1\frac{\use_i:nn#2}{\use_ii:nn#2}\rule[-6pt]{0pt}{0pt}#3}}
	}
	{\regex_match:nnTF {NoValue}{\use_ii:n#1}
		{\,\mathrm{e}^{\mkern-2mu\rule[-6pt]{0pt}{0pt}\use_i:nn#1}}
		{\,\mathrm{e}^{\mkern-2mu\frac{\use_i:nn#1}{\use_ii:nn#1}\rule[-6pt]{0pt}{0pt}\use_i:nn#2}}
}}
\ExplSyntaxOff

% uint function \uint[Interval]{function}{differential} (interval optional)
\ExplSyntaxOn
\NewDocumentCommand{\uint}{>{\SplitArgument{1}{,}}D[]{-,-} m m}{%
	\ifthenelse{\equal{\use_i:nn#1}{-}}
	{\int#2\, \dd#3}							% case 1: indefinite integral
	{\regex_match:nnTF {NoValue}{\use_ii:n#1}
		{\ifthenelse{\equal{\use_i:nn#1}{o}}
			{\oint #2\, \dd#3}					% case 2: Closed loop Integral
			{\int\c_math_subscript_token{\use_i:nn#1}#2 \dd#3}		% case 3: Volume/Surface Integral
		}
		{\int\limits\c_math_subscript_token{\use_i:nn#1}\c_math_superscript_token{\use_ii:nn#1}#2 \dd#3}	% case 4: Integral with boundaries
	}
}
\ExplSyntaxOff

% redefine cmd to add spacing in math mode
\let\oldtfrac\tfrac
\renewcommand{\tfrac}[2]{\hspace{1pt}\oldtfrac{#1}{#2}\hspace{1pt}}
\let\oldunit\unit
\renewcommand{\unit}[1]{\hspace{4pt}\oldunit{#1}}

\hypersetup{%
	colorlinks=true,
	breaklinks=true,
	linkcolor=blue,
	urlcolor=blue,
	citecolor=blue,
	linkbordercolor={0 0 1}
}


%\setlength{\parindent}{0.0in}
%\setlength{\parskip}{0.5in}

% define customizeable variables
%\newcommand*{\name}[1]{\def\@name{#1}}


% create header
\clearpairofpagestyles
\setkomafont{pageheadfoot}{\normalfont}
%\ihead{\@name}
\chead{}
\ohead{\pagemark}
\headheight 0pt
\headsep 2.5em
\footheight 45pt
 
