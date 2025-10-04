# Copilot Instructions (repository-wide, language-agnostic)

These instructions guide GitHub Copilot to generate changes consistent with this repository's conventions, regardless of programming language.

## Glossary

- **Tunables**: user-adjustable parameters that shape behavior, exposed via options or configuration files.
- **Canonical defaults**: the single, authoritative definition of all tunables and their defaults.

## Core principles

- **Design patterns**: prefer established patterns (e.g., factory, singleton, strategy) for code organization and extensibility.
- **Algorithmic**: prefer algorithms or heuristics minimizing time and space complexity.
- **DRY**: avoid duplication of logic, data, and naming. Factor out commonalities.
- **Single source of truth**: maintain a canonical defaults map for configuration tunables. Derive all user-facing options automatically.
- **Naming coherence**: prefer semantically accurate names across code, documentation, directories, and outputs. Avoid synonyms that create ambiguity.
- **English-only**: code, tests, logs, comments, and documentation must be in English.
- **Small, verifiable changes**: prefer minimal diffs that keep public behavior stable unless explicitly requested.
- **Tests-first mindset**: add or update minimal tests before refactoring or feature changes.

## Options and configuration

- **Dynamic generation**: derive CLI and configuration options automatically from canonical defaults. Avoid manual duplication.
- **Merge precedence**: defaults < user options < explicit overrides (highest precedence). Never silently drop user-provided values.
- **Validation**: enforce constraints (choices, ranges, types) at the option layer with explicit typing.
- **Help text**: provide concrete examples for complex options, especially override mechanisms.

## Statistical conventions

- **Hypothesis testing**: use a single test statistic (e.g., t-test) when possible.
- **Divergence metrics**: document direction explicitly (e.g., KL(A||B) vs KL(B||A)); normalize distributions; add numerical stability measures.
- **Distance vs divergence**: distinguish clearly; use consistent terminology.
- **Effect sizes**: report alongside test statistics and p-values; use standard formulas; document directional interpretation.
- **Distribution comparisons**: use multiple complementary metrics (parametric and non-parametric).
- **Correlations**: prefer robust estimators; report the correlation estimate with a confidence interval (parametric or bootstrap) when feasible.
- **Uncertainty quantification**: use confidence intervals or credible intervals methods when feasible.
- **Normality tests**: combine visual diagnostics (e.g., QQ plots) with formal tests when assumptions matter.
- **Multiple testing**: document corrections or acknowledge their absence.

## Simulation and data generation

- **Realism**: correlate synthetic outcomes with causal factors to reflect plausible behavior.
- **Reproducibility**: use explicit random seeds; document them.
- **Edge cases**: test empty datasets, constants, extreme outliers, boundary conditions.
- **Numerical safety**: guard against zero/NaN/Inf; validate intermediate results.

## Reporting conventions

- **Structure**: start with run configuration, then stable section order for comparability.
- **Format**: use structured formats (e.g., tables) for metrics; avoid free-form text for data.
- **Interpretation**: include threshold guidelines; avoid overclaiming certainty.
- **Artifacts**: timestamp outputs; include configuration metadata.

## Implementation guidance for Copilot

- **Before coding**:
  - Locate and analyze thoroughly relevant existing code.
  - Identify code architecture and design patterns.
  - Identify canonical defaults and naming patterns.
- **When adding a tunable**:
  - Add to defaults with safe value.
  - Update documentation and serialization.
- **When implementing analytical methods**:
  - Follow statistical conventions above.
- **When refactoring**:
  - Keep APIs stable; provide aliases if renaming.
  - Update code, tests, and docs atomically.

## Quality gates

- Build/lint/type checks pass (where applicable).
- Tests pass (where applicable).
- Documentation updated to reflect changes.
- Logs use appropriate levels (error, warn, info, debug).
- PR title and commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) format.

## Examples

### Naming coherence

**Good** (consistent style, clear semantics):

```python
threshold_value = 0.06
processing_mode = "piecewise"
```

**Bad** (mixed styles, ambiguous):

```python
thresholdValue = 0.06    # inconsistent case style
threshold_aim = 0.06     # synonym creates ambiguity
```

### Dynamic option generation

```python
DEFAULT_PARAMS = {
    "threshold_value": 0.06,
    "processing_mode": "piecewise",
}

def add_cli_options(parser):
    for key, value in DEFAULT_PARAMS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
```

### Statistical reporting

```markdown
| Metric      | Value | Interpretation        |
| ----------- | ----- | --------------------- |
| KL(A‖B)     | 0.023 | < 0.1: low divergence |
| Effect size | 0.12  | small to medium       |
```

---

By following these instructions, Copilot should propose changes that are consistent and maintainable across languages.
