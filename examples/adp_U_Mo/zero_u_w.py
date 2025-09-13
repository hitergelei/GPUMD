#!/usr/bin/env python3
"""
Zero out u2r and w2r tables in an ADP (setfl-like) potential file.

Usage:
  python zero_u_w.py [input_file] [output_file]

If not provided, defaults to:
  input_file  = U_Mo.alloy.adp_cor.txt
  output_file = U_Mo.alloy.adp_cor_nowuw.txt

The script preserves the original formatting and line breaks while
replacing the u2r and w2r sections with zeros (keeping the same token counts).
"""

from pathlib import Path
import sys


def read_tokens_count(line: str) -> int:
    return 0 if not line.strip() else len(line.split())


def consume_block(lines, start_idx, n_values):
    """Consume lines from start_idx until we accumulate n_values tokens.
    Returns (end_idx (exclusive), captured_lines, token_counts_per_line).
    """
    cur = start_idx
    cnt = 0
    captured = []
    counts = []
    while cnt < n_values:
        if cur >= len(lines):
            raise RuntimeError("Unexpected EOF when consuming a data block.")
        line = lines[cur]
        n = read_tokens_count(line)
        cnt += n
        captured.append(line)
        counts.append(n)
        cur += 1
    if cnt != n_values:
        # setfl allows last line not exact multiple; this ensures exact count
        # If more tokens were found than expected, the file is malformed for our expectations.
        pass
    return cur, captured, counts


def main():
    if len(sys.argv) >= 2:
        infile = Path(sys.argv[1])
    else:
        infile = Path('U_Mo.alloy.adp_cor.txt')
    if len(sys.argv) >= 3:
        outfile = Path(sys.argv[2])
    else:
        stem = infile.stem
        suffix = infile.suffix
        outfile = infile.with_name(stem + '_nowuw' + suffix)

    lines = infile.read_text().splitlines()
    if len(lines) < 6:
        raise RuntimeError('File too short to be a valid ADP setfl file.')

    out_lines = []

    # Header: first 3 comment lines
    out_lines.extend(lines[:3])

    # Line 4: Nelements Element1 Element2 ...
    header4 = lines[3].split()
    try:
        nelems = int(header4[0])
    except Exception as e:
        raise RuntimeError('Failed to parse number of elements on line 4.')
    if len(header4) != 1 + nelems:
        # Some files may swap line 4/5 meaning; this script assumes the common layout
        # used by U_Mo.alloy.adp_cor.txt
        pass
    out_lines.append(lines[3])

    # Line 5: nrho drho nr dr rc
    header5 = lines[4].split()
    if len(header5) < 5:
        raise RuntimeError('Failed to parse line 5: expected nrho drho nr dr rc')
    try:
        nrho = int(float(header5[0]))
        drho = float(header5[1])
        nr = int(float(header5[2]))
        dr = float(header5[3])
        rc = float(header5[4])
    except Exception:
        raise RuntimeError('Failed to parse numeric values on line 5.')
    out_lines.append(lines[4])

    cur = 5
    # For each element: one info line, Frho (nrho values), rhor (nr values)
    elem_info_lines = []
    for _ in range(nelems):
        # info line
        elem_info_lines.append(lines[cur])
        cur += 1
        # Frho
        cur, frho_block, _ = consume_block(lines, cur, nrho)
        # rhor
        cur, rhor_block, _ = consume_block(lines, cur, nr)
        out_lines.extend([elem_info_lines[-1]])
        out_lines.extend(frho_block)
        out_lines.extend(rhor_block)

    # Pairs count
    npairs = nelems * (nelems + 1) // 2

    # z2r block: keep as-is
    for _ in range(npairs):
        cur, z2r_block, _ = consume_block(lines, cur, nr)
        out_lines.extend(z2r_block)

    # u2r block: replace with zeros matching original token counts per line
    u2r_zero_lines = []
    for _ in range(npairs):
        # capture to get original line structure
        nxt, u2r_block, counts = consume_block(lines, cur, nr)
        cur = nxt
        for k, line in enumerate(u2r_block):
            n = counts[k]
            if n == 0:
                u2r_zero_lines.append(line)
            else:
                u2r_zero_lines.append(' '.join(['0'] * n))
    out_lines.extend(u2r_zero_lines)

    # w2r block: replace with zeros matching original token counts per line
    w2r_zero_lines = []
    for _ in range(npairs):
        nxt, w2r_block, counts = consume_block(lines, cur, nr)
        cur = nxt
        for k, line in enumerate(w2r_block):
            n = counts[k]
            if n == 0:
                w2r_zero_lines.append(line)
            else:
                w2r_zero_lines.append(' '.join(['0'] * n))
    out_lines.extend(w2r_zero_lines)

    # Write output
    outfile.write_text('\n'.join(out_lines) + '\n')
    print(f"Wrote: {outfile} (u2r/w2r set to zero).")


if __name__ == '__main__':
    main()

