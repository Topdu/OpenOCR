#!/usr/bin/env python3
"""Post-process display-formula LaTeX in prediction .md files.

Drop-in step for the inference pipeline: predict -> .md -> THIS -> evaluate.

Applies only *general, safe* formatting rules that fix unambiguous LaTeX defects
without altering valid math. Scope is deliberately narrow so nothing outside a
genuine display formula is ever touched:

  * Only whole-line display spans are rewritten -- a line whose stripped content
    is exactly ``$$...$$`` or ``\\[...\\]``.
  * Inline ``$...$`` is NOT touched (a lone ``$`` is currency in prose, so a
    greedy match would span unrelated text and corrupt it).
  * Prose / table lines that merely contain a stray or unbalanced ``$`` are left
    alone.

Rules applied to a display formula's inner content:
  1. escape a bare ``%``            -> ``\\%``   (unescaped % is a LaTeX comment
                                                  that silently eats the line)
  2. drop a stray ``$``             -> removed   (display content must not hold $)
  3. fullwidth parens ``（ ）``      -> ``( )``
  4. fullwidth ``， ；`` -> ``, ;`` in math context only: skipped inside
     ``\\text{}``-like spans, and skipped entirely when a fullwidth period ``。``
     is present (signals natural-language / handwriting content).
  5. drop a bare ``&`` outside any alignment environment (compile-error only).
  6. map raw Unicode math glyphs (``∑ ≤ → α``) to LaTeX commands, outside
     ``\\text{}`` spans -- identity on formulas that already use LaTeX commands.
  7. balance group braces (rescues unmatched ``{ }`` compile failures only).

Usage:
    # rewrite a directory in place
    python tools/postprocess_formula_md.py <md_dir>

    # write results to a separate directory (leaves the input untouched)
    python tools/postprocess_formula_md.py <src_dir> --out <dst_dir>

    # preview only: report files that would change, write nothing
    python tools/postprocess_formula_md.py <md_dir> --dry-run
"""
import argparse
import os
import re
import shutil
import sys

_TEXT_MACROS = (
    '\\text{', '\\mathrm{', '\\mathbf{', '\\mathit{',
    '\\textbf{', '\\mbox{', '\\hbox{', '\\textrm{',
)


def _protect_text_spans(s):
    """Mask ``\\text{}``-like spans so punctuation rules skip natural-language text."""
    spans = []
    out = []
    i, n = 0, len(s)
    while i < n:
        matched = None
        for mac in _TEXT_MACROS:
            if s.startswith(mac, i):
                matched = mac
                break
        if matched:
            j = i + len(matched)
            depth = 1
            while j < n and depth > 0:
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                j += 1
            spans.append(s[i:j])
            out.append('\x00%d\x00' % (len(spans) - 1))
            i = j
        else:
            out.append(s[i])
            i += 1
    return ''.join(out), spans


def _restore(s, spans):
    for idx, sp in enumerate(spans):
        s = s.replace('\x00%d\x00' % idx, sp)
    return s


# Raw Unicode math symbols the model sometimes emits instead of LaTeX commands
# (e.g. "∑", "≤", "→", "α"). Each maps to an unambiguous LaTeX command. A
# trailing space keeps a control word from swallowing the next token (\alpha x).
# Values are chosen so the rendered glyph is identical, hence CDM-neutral, while
# the string becomes valid LaTeX (edit_dist drops toward the reference).
_UNICODE_TO_LATEX = {
    '∑': r'\sum ', '∏': r'\prod ', '∫': r'\int ', '∮': r'\oint ',
    '√': r'\sqrt ', '±': r'\pm ', '∓': r'\mp ', '×': r'\times ',
    '÷': r'\div ', '·': r'\cdot ', '∗': r'\ast ',
    '≤': r'\leq ', '≥': r'\geq ', '≠': r'\neq ', '≈': r'\approx ',
    '≡': r'\equiv ', '≪': r'\ll ', '≫': r'\gg ', '∝': r'\propto ',
    '→': r'\rightarrow ', '←': r'\leftarrow ', '↔': r'\leftrightarrow ',
    '⇒': r'\Rightarrow ', '⇐': r'\Leftarrow ', '⇔': r'\Leftrightarrow ',
    '⟶': r'\longrightarrow ', '↦': r'\mapsto ', '∞': r'\infty ',
    '∂': r'\partial ', '∇': r'\nabla ', '∈': r'\in ', '∉': r'\notin ',
    '∋': r'\ni ', '⊂': r'\subset ', '⊃': r'\supset ', '⊆': r'\subseteq ',
    '⊇': r'\supseteq ', '∪': r'\cup ', '∩': r'\cap ', '∅': r'\emptyset ',
    '∀': r'\forall ', '∃': r'\exists ', '¬': r'\neg ', '∧': r'\wedge ',
    '∨': r'\vee ', '⊕': r'\oplus ', '⊗': r'\otimes ', '⊥': r'\perp ',
    '∥': r'\parallel ', '∠': r'\angle ', '△': r'\triangle ',
    '∼': r'\sim ', '≅': r'\cong ', '≜': r'\triangleq ',
    'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'δ': r'\delta ',
    'ε': r'\varepsilon ', 'ζ': r'\zeta ', 'η': r'\eta ', 'θ': r'\theta ',
    'ι': r'\iota ', 'κ': r'\kappa ', 'λ': r'\lambda ', 'μ': r'\mu ',
    'ν': r'\nu ', 'ξ': r'\xi ', 'π': r'\pi ', 'ρ': r'\rho ',
    'σ': r'\sigma ', 'τ': r'\tau ', 'υ': r'\upsilon ', 'φ': r'\phi ',
    'χ': r'\chi ', 'ψ': r'\psi ', 'ω': r'\omega ',
    'Γ': r'\Gamma ', 'Δ': r'\Delta ', 'Θ': r'\Theta ', 'Λ': r'\Lambda ',
    'Ξ': r'\Xi ', 'Π': r'\Pi ', 'Σ': r'\Sigma ', 'Φ': r'\Phi ',
    'Ψ': r'\Psi ', 'Ω': r'\Omega ',
}
_UNICODE_MATH_RE = re.compile('|'.join(re.escape(c) for c in _UNICODE_TO_LATEX))


def unicode_to_latex(s: str) -> str:
    """Replace raw Unicode math symbols with their LaTeX commands.

    Runs only on display-formula inner content, and only OUTSIDE ``\\text{}``-like
    spans (where Unicode is legitimate prose, e.g. ``μl`` or Chinese units). It is
    an identity on formulas that already use LaTeX commands, so it never regresses
    a correct formula; it only rescues ones the model emitted as raw glyphs.
    """
    if not _UNICODE_MATH_RE.search(s):
        return s
    protected, spans = _protect_text_spans(s)
    protected = _UNICODE_MATH_RE.sub(lambda m: _UNICODE_TO_LATEX[m.group(0)], protected)
    return _restore(protected, spans)


# Alignment environments in which a bare '&' is legal (column/row separator).
# Detected by name so a `&` is only ever removed when NO such env is open.
_ALIGN_ENVS = (
    'array', 'align', 'aligned', 'alignat', 'alignedat', 'matrix',
    'pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix', 'smallmatrix',
    'cases', 'dcases', 'split', 'gather', 'gathered', 'multline',
    'eqnarray', 'subarray', 'tabular',
)
_HAS_ALIGN_ENV = re.compile(
    r'\\begin\{(?:' + '|'.join(_ALIGN_ENVS) + r')\*?\}'
)


def balance_braces(s: str) -> str:
    """Make LaTeX group braces balanced.

    Drops a '}' that has no matching '{', then appends any '{' left unclosed.
    Escaped braces (\\{ \\}) are literals and ignored. This is an IDENTITY on any
    already-balanced string, so it can only alter a formula that is already a
    LaTeX compile failure (unbalanced braces render to nothing -> CDM 0); it can
    never regress a formula that compiled.
    """
    out = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            # keep escape sequence (incl. \{ and \}) verbatim, uncounted
            out.append(c)
            out.append(s[i + 1])
            i += 2
            continue
        if c == '{':
            depth += 1
            out.append(c)
        elif c == '}':
            if depth > 0:
                depth -= 1
                out.append(c)
            # else: stray close brace -> drop
        else:
            out.append(c)
        i += 1
    if depth > 0:
        out.append('}' * depth)
    return ''.join(out)


def strip_stray_amp(s: str) -> str:
    """Remove bare '&' when no alignment environment is present.

    A bare '&' (not the escaped literal \\&) is only legal inside an alignment
    environment (array/align/matrix/cases/...). When the formula opens none of
    them, every bare '&' is already a compile error, so removing it cannot
    regress a formula that compiled. Formulas that DO contain such an env are
    left completely untouched.
    """
    if _HAS_ALIGN_ENV.search(s):
        return s
    if '&' not in s:
        return s
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            out.append(c)
            out.append(s[i + 1])  # preserve \& literal
            i += 2
            continue
        if c == '&':
            i += 1  # drop stray alignment marker
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def apply_safe_rules(inner: str) -> str:
    """Apply general safe formatting rules to a display formula's inner content."""
    if not inner:
        return inner
    # Rule 1: escape bare '%' (LaTeX comment char that silently eats the line).
    inner = re.sub(r'(?<!\\)%', r'\\%', inner)
    # Rule 2: drop stray '$' left inside display-math content.
    inner = inner.replace('$', '')
    # Rule 3: fullwidth parens -> halfwidth (structural, safe even inside text).
    inner = inner.replace('（', '(').replace('）', ')')
    # Rule 4: fullwidth , ; -> halfwidth, only in a math context.
    if '。' not in inner:
        protected, spans = _protect_text_spans(inner)
        protected = protected.replace('，', ',').replace('；', ';')
        inner = _restore(protected, spans)
    # Rule 5: drop bare '&' left outside any alignment environment (compile-error
    # only; identity on formulas that use array/align/matrix/cases/...).
    inner = strip_stray_amp(inner)
    # Rule 6: map raw Unicode math glyphs (∑ ≤ → α ...) to LaTeX commands, outside
    # \text{} spans. Identity on formulas already using LaTeX commands.
    inner = unicode_to_latex(inner)
    # Rule 7: balance group braces (identity on already-balanced formulas; only
    # rescues formulas that fail to compile from unmatched { }).
    inner = balance_braces(inner)
    return inner


# A genuine display formula stands alone on its own line: the whole stripped line
# is `$$...$$` or `\[...\]`. Only such whole-line spans are rewritten.
_WHOLE_LINE_RE = re.compile(r'^(\s*)(\$\$)(.*)(\$\$)(\s*)$|^(\s*)(\\\[)(.*)(\\\])(\s*)$')


def rewrite_text(text: str) -> str:
    """Rewrite every whole-line display formula in a markdown document."""
    out_lines = []
    for line in text.split('\n'):
        m = _WHOLE_LINE_RE.match(line)
        # guard against a line holding two `$$...$$` (inner still contains `$$`)
        if m and '$$' not in (m.group(3) or (m.group(8) or '')):
            if m.group(2) is not None:  # $$ ... $$
                lead, od, inner, cd, trail = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            else:                        # \[ ... \]
                lead, od, inner, cd, trail = m.group(6), m.group(7), m.group(8), m.group(9), m.group(10)
            out_lines.append(lead + od + apply_safe_rules(inner) + cd + trail)
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)


def process_file(src_path: str, dst_path: str) -> bool:
    """Rewrite one file. Returns True if content changed."""
    with open(src_path, 'r', encoding='utf-8') as f:
        text = f.read()
    fixed = rewrite_text(text)
    changed = fixed != text
    if dst_path != src_path or changed:
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
    return changed


def main():
    ap = argparse.ArgumentParser(description='Safe LaTeX post-processing for prediction .md files.')
    ap.add_argument('src_dir', help='directory of prediction .md files')
    ap.add_argument('--out', help='write to this directory instead of editing in place')
    ap.add_argument('--dry-run', action='store_true', help='report changes without writing')
    args = ap.parse_args()

    src_dir = args.src_dir
    if not os.path.isdir(src_dir):
        sys.exit(f'not a directory: {src_dir}')

    dst_dir = args.out or src_dir
    if args.out and not args.dry_run:
        # mirror the whole tree so non-.md files come along too
        if os.path.abspath(args.out) != os.path.abspath(src_dir):
            shutil.copytree(src_dir, args.out, dirs_exist_ok=True)

    total = changed = 0
    for name in os.listdir(src_dir):
        if not name.endswith('.md'):
            continue
        total += 1
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if args.dry_run:
            with open(src, 'r', encoding='utf-8') as f:
                text = f.read()
            if rewrite_text(text) != text:
                changed += 1
                print('would modify:', name)
        else:
            if process_file(src, dst):
                changed += 1

    verb = 'would modify' if args.dry_run else 'modified'
    print(f'processed {total} .md files, {verb} {changed}')


if __name__ == '__main__':
    main()
