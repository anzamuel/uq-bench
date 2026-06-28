= `uq_bench.metrics`

Targets $y_i$ with prediction interval $[ell_i, u_i]$ over $N$ test points at miscoverage level $alpha$, with lower and upper quantile levels $alpha\/2$ and $1 - alpha\/2$.

== `picp` <picp>

Prediction interval coverage probability, the fraction of targets inside the interval.

$ "PICP" = 1/N sum_(i=1)^N bb(1){ell_i <= y_i <= u_i} $

== `mpiw` <mpiw>

Mean prediction interval width.

$ "MPIW" = 1/N sum_(i=1)^N (u_i - ell_i) $

== `niw` <niw>

Mean interval width normalized by the target range.

$ "NIW" = "MPIW" / (max_i y_i - min_i y_i) $

== `pinball` <pinball>

Average pinball loss of the two endpoints, with the pinball loss at level $tau$ defined as

$ "QL"_tau (y, q) = (y - q) (tau - bb(1){y <= q}), $

so that

$ "PINBALL" = 1/(2 N) sum_(i=1)^N [ "QL"_(alpha\/2)(y_i, ell_i) + "QL"_(1 - alpha\/2)(y_i, u_i) ]. $

== `aisl` <aisl>

Average interval score loss, the width plus a $2\/alpha$ penalty for each target outside the interval.

$ "AISL" = 1/N sum_(i=1)^N [ (u_i - ell_i) + 2/alpha (ell_i - y_i) bb(1){y_i < ell_i} + 2/alpha (y_i - u_i) bb(1){y_i > u_i} ] $

== `nciw` <nciw>

Normalized interval width after rescaling each half-width around the center $f_i$ by the smallest factor that reaches coverage $1 - alpha$. The per-point covering scale is

$ c_i = cases(
  (f_i - y_i) / (f_i - ell_i) & "if" y_i < f_i,
  (y_i - f_i) / (u_i - f_i) & "if" y_i > f_i,
  0 & "otherwise",
) $

and with $c^*$ the $ceil((1 - alpha) N)$-th smallest $c_i$,

$ "NCIW" = c^* dot "NIW". $
