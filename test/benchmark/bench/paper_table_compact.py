"""Compact Table 1: one column per tool and a reduced row set, from the full benchmark table.

    python paper_table_compact.py tables/benchmark_models.tex tables/benchmark_models_compact.tex

Each column is one model. Tool names are abbreviated (key in the table caption) and set on two
horizontal lines with the model below the tool; columns take their natural width and the
spare width is spread between them, so nine columns fit the text width at the body font size. The
5mC sub-groups are merged, three rows are dropped (listed in DROP), the "vs."
control names are replaced by a marker on every matched (two-sample) row, and the
dagger/double-dagger marks are removed. The summary rows (mean, worst, parameters)
are copied from the full table unchanged, so they still cover all 15 rows.
"""
import re
import sys

# (header text as written in the full table, group, header line 1, header line 2) in output order.
# Tool names are abbreviated (key in the table caption); the second line names the model.
ALL_COLUMNS = [
    (r'\sysname', 'ours', r'\sysname', ''),
    (r'Dorado\\4mC+5mC', 'released', 'Dor.', '4mC+5mC'),
    (r'Dorado\\5mC+5hmC', 'released', 'Dor.', '5(h)mC'),
    (r'Dorado\\5mCG+5hmCG', 'released', 'Dor.', '5(h)mCG'),
    (r'Dorado\\6mA', 'released', 'Dor.', '6mA'),
    (r'UniMeth\\5mC', 'released', 'UM', '5mC'),
    (r'UniMeth\\6mA', 'released', 'UM', '6mA'),
    (r'Rockfish\\5mCG', 'released', 'RF', '5mCG'),
    (r'UniMeth-FT$^{\dagger}$\\4mC', 'finetuned', 'UM-FT', '4mC'),
    (r'UniMeth-FT$^{\dagger}$\\5hmC', 'finetuned', 'UM-FT', '5hmC'),
    (r'UniMeth-FT$^{\dagger}$\\5hmU', 'finetuned', 'UM-FT', '5hmU'),
    (r'UniMeth-Agn$^{\dagger}$\\all', 'finetuned', 'UM-Agn', 'all'),
    (r'Uncalled4$^{*}$\\vs.\ control', 'comparative', 'Unc4', 'control'),
    (r'NovaMod$^{*}$\\own control', 'comparative', 'NM', 'control'),
]
# models shown in Table 1 (line-2 labels per tool); the rest stay in the full supplementary table
SHOW = {'Dor.': ['4mC+5mC', '6mA'], 'UM': ['5mC'], 'RF': ['5mCG'],
        'UM-FT': ['5hmU'], 'UM-Agn': ['all'], 'Unc4': ['control'], 'NM': ['control']}
COLUMNS = [c for c in ALL_COLUMNS if c[1] == 'ours' or c[3] in SHOW.get(c[2], [])]
GROUPS = [('released', 'Released'), ('finetuned', 'Fine-tuned'), ('comparative', 'Comparative')]
HEADFONT = r'\scriptsize\bfseries'   # two short header lines, set horizontally
GROUPFONT = r'\footnotesize\itshape'
MATCHED = r'$^{\ast}$'          # marker of rows scored against a matched unmodified library
DROP = ['ONT Benchmark -- 5mC', 'M.SssI', 'ONT Benchmark -- 6mA']   # substrings of row labels to drop
MERGE_GROUPS = {'5mC, CpG': '5mC', '5mC, non-CpG': None, '5mC, all contexts': None}   # None: drop the sub-header
# row names in Table 1: scientific names throughout, no motif, enzyme or control (the
# modification group and the supplementary tables carry those); keys match the label
# after the "vs." part is removed
NAMES = {
    r'\textit{A.~thaliana}': r'\textit{A.~thaliana}',
    'HG002': r'\textit{H.~sapiens} (HG002)',
    r'\textit{O.~sativa}': r'\textit{O.~sativa}',
    'Mouse brain': r'\textit{M.~musculus} (brain)',
    r'\textit{E.~coli} M.SssI': r'\textit{E.~coli}',
    r'\textit{E.~coli} Dcm': r'\textit{E.~coli} (non-CpG)',
    r'\textit{E.~coli} Dam': r'\textit{E.~coli}',
    'ONT Benchmark -- 5mC': 'Synthetic constructs',
    'ONT Benchmark -- 5hmC': 'Synthetic constructs',
    'ONT Benchmark -- 6mA': 'Synthetic constructs',
    r'\textit{Anabaena}': r'\textit{Anabaena}',
    r'\textit{H.~pylori} 26695': r'\textit{H.~pylori} 26695',
    r'\phage native': r'Phage \phage',
}
# context added to rows that lose their sub-header
CONTEXT = {('5mC, non-CpG', r'\textit{A.~thaliana}'): 'non-CpG'}


def header_text(cell):
    m = re.search(r'\\shortstack\{(.*?)\}\}', cell)
    return m.group(1) if m else cell.strip().replace(r'\textbf{', '').rstrip('}')


def relabel(label, group):
    """Row label without 'vs. <control>' or motif, renamed per NAMES, with the matched marker."""
    lab = label.strip().replace(r'$^{\ddagger}$', '')
    indent = re.match(r'\\hspace\{[^}]*\}', lab)
    indent, lab = (indent.group(0), lab[indent.end():]) if indent else ('', lab)
    matched = ' vs.\\' in lab or ' vs. ' in lab
    circ = r'$^{\circ}$' in lab
    lab = lab.replace(r'$^{\circ}$', '')
    lab = re.sub(r'\s+vs\.\\?\s.*$', '', lab).strip()
    lab = NAMES.get(lab, lab)
    for (g, key), ctx in CONTEXT.items():
        if g == group and key in lab:
            lab = lab + ' (' + ctx + ')'
    if circ:
        lab = lab + r'$^{\circ}$'
    if matched:
        lab = '{' + lab + '}' + MATCHED      # braces: no \xspace gap before the marker
    return indent + lab


def main(src, dst):
    lines = [l for l in open(src).read().splitlines() if not l.startswith('%')]
    head = next(l for l in lines if l.startswith(r'\textbf{Dataset}'))
    cells = head.rstrip()[:-2].split('&')
    texts = [header_text(c) for c in cells]
    idx = [0] + [texts.index(t) for t, *_ in COLUMNS]
    ncol = len(idx)
    out = ['% GENERATED by test/benchmark/bench/paper_table_compact.py from ' + src.split('/')[-1]
           + '; abbreviated two-line headers, reduced rows.']
    group = None
    for l in lines:
        if l.startswith(r'\setlength{\tabcolsep}'):
            out.append(r'\setlength{\tabcolsep}{1pt}%')
        elif l.startswith(r'\begin{tabular}'):
            out.append(r'\begin{tabular*}{\tablewidth}{@{\extracolsep{\fill}}l *{%d}{c}@{}}' % (ncol - 1))
        elif l.startswith(r'\end{tabular}'):
            out.append(r'\end{tabular*}')
        elif l.startswith(r'\multicolumn{'):          # modification group row
            m = re.search(r'\\textbf\{(.*?)\}', l)
            group = m.group(1)
            name = MERGE_GROUPS.get(group, group)
            if name is None:
                continue
            out.append(r'\multicolumn{%d}{@{}l}{\textbf{%s}} \\' % (ncol, name))
        elif l.startswith(r'& \multicolumn{'):        # tool group header -> new groups
            parts, rules, col = [], [], 3            # column 1 = dataset, 2 = ours
            for key, title in GROUPS:
                n = sum(1 for _, g, *_ in COLUMNS if g == key)
                if not n:
                    continue
                parts.append(r'\multicolumn{%d}{c}{%s %s}' % (n, GROUPFONT, title))
                rules.append(r'\cmidrule(lr){%d-%d}' % (col, col + n - 1))
                col += n
            out.append('& & ' + ' & '.join(parts) + r' \\')
            out.append(''.join(rules))
        elif l.startswith(r'\cmidrule'):
            continue
        elif '&' in l and l.rstrip().endswith('\\\\'):
            c = l.rstrip()[:-2].split('&')
            if len(c) != len(cells):
                out.append(l); continue
            sel = [c[i].strip() for i in idx]
            if l.startswith(r'\textbf{Dataset}'):
                sel = [sel[0]] + [r'\multicolumn{1}{c}{%s\begin{tabular}[b]{@{}c@{}}%s\\%s\end{tabular}}'
                                  % (HEADFONT, a, b) if b else r'\multicolumn{1}{c}{%s %s}' % (HEADFONT, a)
                                  for _, _, a, b in COLUMNS]
            elif l.startswith(r'\hspace'):
                if any(d in sel[0] for d in DROP):
                    continue
                sel[0] = relabel(sel[0], group)
            out.append(' & '.join(sel) + r' \\')
        else:
            out.append(l)
    out = [l.replace(r'\textbf{Mean over the five modifications}', r'\textbf{Mean over 5 types}').replace('Worst modification &', 'Worst type &')
           .replace(r'\hspace{3pt}', r'\hspace{2pt}') for l in out]
    # parameter counts to two significant figures: the full-precision cells were the widest in their columns
    out = [re.sub(r'\b100\.3M', '100M', re.sub(r'\b34\.7M', '35M', re.sub(r'\b1\.80M', '1.8M', l)))
           if l.startswith('Parameters') else l for l in out]
    open(dst, 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
