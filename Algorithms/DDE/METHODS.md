# Estimation developments and implementation

This note describes the published source, rather than asserting that the adaptive estimation algorithm has converged. The fixed-parameter latent transition targets the rank-constrained posterior after warmup; parameter optimization, thresholding and changing active dimensions make the overall estimation trajectory different from a stationary posterior chain.

## Binary update, tempering and conditional-product warmup

`cmv_binary` visits the top binary layer before the bottom. Each observation gets a state-independent random permutation; blocks are updated immediately. Candidate configurations are enumerated and normalized from their joint conditional log probabilities, including the Gaussian contribution for the bottom layer, the logistic layer and the top Bernoulli prior where applicable. With the current `vi_layers=[false false]`, no variational update is selected.

Inverse temperature beta multiplies each candidate log weight. CP uses `bl_mix` to mix the enumerated joint-block conditional with a product of single-site conditionals evaluated at the same OLD block state. It selects one of these block laws before drawing. The product branch is a warmup device and is not assumed to preserve the ordinary joint target. Lambda=1 removes it. Beta=1 removes tempering. The current CP ramp is `min(1,(iteration-1)/59)`; the original beta curve is stretched to reach one at iteration 30. Arms without a warmup device use its final value from the start. After both endpoints, updates use the ordinary joint block conditional. Separate rows can be processed together because they are conditionally independent; dependent blocks and rank groups must use their newly updated neighbors.

## Rank-group Gibbs

For a fixed Gaussian column, let `mu_i` and variance `gamma` be determined by the current binary factors and parameters. Observed ties partition rows into ordered groups. In increasing group order, sample every row in a group from its Gaussian truncated between the current previous-group maximum and next-group minimum. Within a group there is no rank order. Immediately use the new values for subsequent group bounds. `dde_rank_gibbs` performs one such sweep per Z arm iteration and verifies strict between-group inequalities. Its rejection samplers avoid clipping Gaussian CDF probabilities. It handles missing ranks with unconstrained draws; the paper generator supplies complete ranks. The collapsed-maxima implementation should not be assumed to share that missing-rank behavior.

## Collapsed maxima: derivation

For group l with independent Gaussian densities f_i and CDFs F_i, define

\[
q_i(a,b)=F_i(b)-F_i(a),\qquad Q_l(a,b)=\prod_{i\in G_l}q_i(a,b).
\]

For every nonfinal group, retain only its maximum M_l while integrating its interiors. Differentiating Q_l with respect to its upper limit gives the maximum factor

\[
h_l(a,b)=\frac{\partial Q_l(a,b)}{\partial b}
 =\sum_{i\in G_l} f_i(b)\prod_{j\in G_l,\,j\ne i}q_j(a,b)
 =Q_l(a,b)\sum_{i\in G_l}\frac{f_i(b)}{q_i(a,b)}.
\]

Each summand identifies the unique maximum-attaining observation (ties have probability zero). With M_0=-infinity and L ordered groups, the collapsed target on M_1<...<M_(L-1) is proportional to

\[
\prod_{l=1}^{L-1}h_l(M_{l-1},M_l)\;Q_L(M_{L-1},+\infty).
\]

An interior maximum x therefore has conditional density proportional to `h_l(M_(l-1),x) h_(l+1)(x,M_(l+1))`. For the last maximum the second factor is the final-group survival Q_L. `cma_logh` evaluates these factors using stable Gaussian log-interval probabilities and log-sum-exp. Repeated means are grouped with multiplicities; winner probabilities for a mean class include that count.

`cma_column` uses scalar slice transitions on these collapsed conditionals. The current estimator requests two sequential sweeps in order 1,...,L-1, each using freshly updated neighboring maxima. This is a Markov transition preserving the collapsed target in exact arithmetic, not an independent joint posterior draw. Finite precision and finite rejection/expansion limits remain numerical qualifications.

After the maxima sweeps, regenerate each nonfinal group: select a winner with probability proportional to f_i(M_l)/q_i(M_(l-1),M_l), place it at M_l, and draw the remaining observations independently from their Gaussians truncated to that interval. Draw the final group above M_(L-1). Grouping equal means uses a count-weighted class draw followed by a uniform winner within the class. A single-category column reduces to unconstrained Gaussian regeneration.

**Do not condition on the old interiors during the collapsed maxima sweep.** Integrating them out is what yields the h factors; retaining them as bounds would change the transition.

```text
For each Gaussian column:
    identify ordered tied-rank groups and current nonfinal maxima
    repeat twice:
        for l = 1,...,L-1:
            slice-update M_l using its two neighboring factors and fresh bounds
    for each nonfinal group:
        draw its maximum winner; fix winner to M_l
        regenerate other rows between M_(l-1) and M_l
    regenerate final group above M_(L-1)
    assert strict rank consistency
```

This explains why the collapsed update can move an entire tied group while retaining the rank-likelihood target. It does not imply superior mixing in every setting; compare stationary conditional expectations and equal-compute diagnostics separately from estimator MSE. The finite-run study shows a sample-size-dependent B1 crossover.

## Layer-specific averaging and dimensions

The Gaussian layer retains moving sufficient statistics G=X'X, H=X'Z and ZZ=sum(Z.^2). The logistic layer retains configuration-indexed weighted counts N and response totals T for A2 configurations; unobserved configurations from the latest draw still decay. The objective uses weighted responses T/N. Top Bernoulli probabilities use averaged success totals divided by sample size. Averaging each layer's likelihood contributions avoids substituting an average latent binary matrix into a nonlinear likelihood. Penalties, CSP updates and active-set handling remain as in the frozen engine.

The initial first-layer width is the largest consecutive initialized-Z covariance eigenvalue ratio plus five, capped by the allowed maximum and usable rank, with a minimum of three. Initial second-layer width is floor(K1/3). CSP uses the original alpha 15/5 and spike 0.05/0.1 settings. The loading threshold is `max(0.3,3*N^(-0.3))`; inactive first-layer rows are gated in B2. These are estimation/selection operations, not evidence of stationary posterior convergence. See `lsa_protocol`, `pfs_arm`, `pcs_csp_update`, `dde_spectral_widths` and the unchanged `lsa_fit` for precise ordering.

The three-method comparison separates H1 versus Z sampler behavior while holding full averaging fixed, and H1 versus H0 averaging behavior under the same collapsed sampler. It does not include raw-objective Gibbs. See [README](README.md) for exact schedules/stopping and [evidence](evidence/README.md) for cohort and interpretation limits.
