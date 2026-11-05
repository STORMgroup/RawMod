"""Fit the wide benchmark tables upright on the one-column text width.

    python -m bench.upright <tables_dir>

The generators write each table with full tool names in the header. This step rewrites them so
that no table needs a landscape page:

  * header tool names are abbreviated and set on two short lines (tool, model), as in Table 1;
  * cells holding two values ("a (b)" or "a / b") are stacked on two lines;
  * the two tables with one column per model (benchmark_models_full, supp_bench_ci) are split
    into two panels, <name>_a.tex and <name>_b.tex, each repeating the RawMod column.

paper_table.py and library_control.py call fit() on what they write; running this module on a
tables directory applies the same step to existing files. Every rewrite is idempotent.
"""
import re
import sys
from pathlib import Path

TOOLS = {
    r'\sysname': r'\sysname',
    'Dorado': 'Dor.',
    'UniMeth': 'UM',
    'UniMeth-FT': 'UM-FT',
    'UniMeth-Agn': 'UM-Agn',
    'DeepMod2': 'DM2',
    'Rockfish': 'RF',
    'MethyNano': 'MN',
    'Uncalled4': 'Unc4',
    'NovaMod': 'NM',
}
MODELS = {'5mC+5hmC': '5(h)mC', '5mCG+5hmCG': '5(h)mCG', r'vs.\ control': 'control', 'own control': 'control'}
HEADFONT = r'\scriptsize\bfseries'

# column panels (1-based data columns of the full table; column 0 is the row label)
PANELS = {
    # RawMod | Dorado x4, UniMeth x2, UniMeth-FT x3 || RawMod | DeepMod2 ... NovaMod
    'benchmark_models_full': ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 11, 12, 13, 14, 15, 16]),
    # Sites, RawMod | Dorado x4, UniMeth x2 || RawMod | UniMeth-FT x3, Rockfish, UniMeth-Agn, Uncalled4, NovaMod
    'supp_bench_ci': ([1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 11, 12, 13, 14, 15]),
}
ABBREVIATE = ['benchmark', 'benchmark_agnostic', 'supp_bench_libcontrol', 'supp_bench_within']
STACK = ['supp_bench_libcontrol', 'supp_bench_within']


def cells(row):
    """Split one tabular row (without its trailing \\\\) at top-level &."""
    out, depth, cur, i = [], 0, '', 0
    while i < len(row):
        c = row[i]
        if c == '\\' and i + 1 < len(row):
            cur += row[i:i + 2]
            i += 2
            continue
        depth += (c == '{') - (c == '}')
        if c == '&' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += c
        i += 1
    return out + [cur]


def is_row(line):
    return line.rstrip().endswith('\\\\') and '&' in line


def is_header(line):
    return bool(re.match(r'\s*\\textbf\{(Dataset|Row)\}\s*&', line))


def unwrap(s, macro):
    m = re.fullmatch(r'\\' + macro + r'\{(.*)\}', s.strip(), re.S)
    return m.group(1) if m else s.strip()


def head_cell(text):
    """Abbreviated two-line header for one tool column; already-abbreviated cells pass through."""
    if HEADFONT in text:
        return text.strip()
    inner = unwrap(unwrap(text, 'textbf'), 'shortstack')
    tool, _, model = inner.partition(r'\\')
    m = re.fullmatch(r'(.*?)(\$\^\{.*\}\$)?', tool.strip())
    name, mark = m.group(1), m.group(2) or ''
    tool = TOOLS.get(name, name) + mark
    model = MODELS.get(model.strip(), model.strip())
    lines = tool + (r'\\' + model if model else '')
    return r'\multicolumn{1}{c}{' + HEADFONT + r'\begin{tabular}[b]{@{}c@{}}' + lines + r'\end{tabular}}'


def abbreviate(tex):
    out = []
    for line in tex.split('\n'):
        if is_header(line):
            c = cells(line.rstrip()[:-2])
            line = ' & '.join([c[0].strip()] + [head_cell(x) for x in c[1:]]) + r' \\'
        out.append(line)
    return '\n'.join(out)


def stack_cell(c):
    s = c.strip()
    m = re.fullmatch(r'(\S+) \((\S+)\)', s) or re.fullmatch(r'(\S+) / (\S+)', s)
    if not m:
        return c
    sep = '(' + m.group(2) + ')' if '(' in s else m.group(2)
    return r' \makecell{' + m.group(1) + r'\\' + sep + '} '


def stack(tex):
    out = []
    for line in tex.split('\n'):
        if is_row(line) and not is_header(line):
            c = cells(line.rstrip()[:-2])
            line = '&'.join([c[0]] + [stack_cell(x) for x in c[1:]]).rstrip() + r' \\'
        out.append(line)
    return '\n'.join(out)


def _groups(line, n):
    """Column -> group label from a '& \\multicolumn{k}{c}{label} & ...' row, for data columns 1..n."""
    g, col = {}, 1
    for c in cells(line.rstrip()[:-2])[1:]:
        m = re.fullmatch(r'\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*', c, re.S)
        k, lab = (int(m.group(1)), m.group(2)) if m else (1, c.strip())
        for j in range(col, col + k):
            g[j] = lab
        col += k
    return g


def panel(tex, keep, tag):
    """Keep data columns `keep` (1-based); the panel tag (a, b) goes in the first header cell."""
    lines = tex.split('\n')
    n = max(len(cells(l.rstrip()[:-2])) for l in lines if is_row(l)) - 1
    groups = None
    out = []
    for line in lines:
        if re.match(r'\s*\\begin\{tabular\}', line):
            out.append(r'\begin{tabular}{@{}l ' + 'c' * len(keep) + '@{}}')
        elif re.match(r'\s*&\s*\\multicolumn', line):
            groups = _groups(line, n)
            runs = []
            for j in keep:
                if runs and runs[-1][0] == groups[j]:
                    runs[-1][1] += 1
                else:
                    runs.append([groups[j], 1])
            out.append(r'\textit{(' + tag + ')} & ' + ' & '.join(
                r'\multicolumn{%d}{c}{%s}' % (k, lab) for lab, k in runs) + r' \\')
            rules, start = [], 2
            for lab, k in runs:
                rules.append(r'\cmidrule(lr){%d-%d}' % (start, start + k - 1))
                start += k
            out.append(''.join(rules))
        elif line.lstrip().startswith(r'\cmidrule') and groups is not None:
            continue
        elif re.match(r'\s*\\multicolumn\{%d\}' % (n + 1), line):
            out.append(re.sub(r'\\multicolumn\{%d\}' % (n + 1), r'\\multicolumn{%d}' % (len(keep) + 1), line, 1))
        elif is_row(line):
            c = cells(line.rstrip()[:-2])
            first = c[0].strip()
            if is_header(line) and groups is None:
                first = r'\textit{(' + tag + r')} ' + first
            out.append(' & '.join([first] + [c[j].strip() for j in keep]) + r' \\')
        else:
            out.append(line)
    return '\n'.join(out)


def fit(path):
    """Apply the upright step for one generated table; returns the files written."""
    path = Path(path)
    name = path.stem
    tex = path.read_text()
    if name in PANELS:
        written = []
        for tag, keep in zip('ab', PANELS[name]):
            p = path.with_name(f'{name}_{tag}.tex')
            p.write_text(panel(abbreviate(tex), keep, tag))
            written.append(p)
        return written
    if name in ABBREVIATE:
        new = abbreviate(tex)
        if name in STACK:
            new = stack(new)
        if new != tex:
            path.write_text(new)
        return [path]
    return []


if __name__ == '__main__':
    d = Path(sys.argv[1])
    for name in list(PANELS) + ABBREVIATE:
        p = d / f'{name}.tex'
        if p.exists():
            for w in fit(p):
                print('wrote', w)
