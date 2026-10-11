"""Shrink a bundled C++ file for judges that limit source size (Codeforces 64 KB, CodeChef 50000 B).

compact(text) -> text, used by showcase/bundle.py. Only drops what the compiler ignores, and
rewrites strings into equal concatenations:
- comments and whitespace (a space is kept wherever two tokens would otherwise merge);
- inline-asm instruction strings "op a, b, c\\n\\t" become macro calls that expand to
  "op a,b,c\\n\\t": QZrrr(QZ3,12,1,0) for three ymm registers, QZorr for a first operand kept
  verbatim (memory, immediate, %[name]), and so on; frequent opcodes get short aliases;
- the repeated clobber list "xmm0", ..., "xmm15" becomes one macro.
Macros are defined before the first line that uses them and undefined after the last.
"""
from __future__ import annotations

import collections
import re

TOKEN = re.compile(r'''
    (?P<str>"(?:\\.|[^"\\\n])*")
  | (?P<chr>'(?:\\.|[^'\\\n])*')
  | (?P<line>//[^\n]*)
  | (?P<block>/\*.*?\*/)
  | (?P<ws>[ \t\r\f\v]+)
  | (?P<nl>\n)
  | (?P<word>[A-Za-z_$][\w$]*|\.?\d(?:[eEpP][+-]|[\w.'])*)
  | (?P<punct>.)
''', re.S | re.X)

WORD = re.compile(r'[\w$]')
OPERATORS = set('+-*/%<>=&|^!:.#')
INSTRUCTION = re.compile(r'^([a-z][a-z0-9]*) (.*)\\n(?:\\t)?$')
YMM = re.compile(r'^%%ymm(\d+)$')
CLOBBERS = ', '.join(f'"xmm{i}"' for i in range(16))


def split_operands(text: str) -> list[str] | None:
    """Operands separated by ', ' outside parentheses; None if any could not be a macro argument."""
    parts, depth, cur = [], 0, ''
    for i, c in enumerate(text):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                return None
        if c == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += c
    parts.append(cur.strip())
    ok = depth == 0 and all(p and ' ' not in p and '"' not in p and '\\' not in p for p in parts)
    return parts if ok else None


class Asm:
    """Rewrites asm instruction strings into macro calls and collects the macros they need."""

    def __init__(self, strings: list[str]):
        self.ops = collections.Counter()
        parsed = [self.parse(s) for s in strings]
        for p in parsed:
            if p:
                self.ops[p[0]] += 1
        self.alias = {op: f'QZ{i}' for i, (op, n) in enumerate(self.ops.most_common()) if n >= 4}
        self.signatures = set()

    @staticmethod
    def parse(literal: str):
        m = INSTRUCTION.match(literal[1:-1])
        if not m:
            return None
        operands = split_operands(m.group(2))
        return (m.group(1), operands) if operands else None

    def rewrite(self, literal: str) -> str:
        p = self.parse(literal)
        if not p:
            return literal
        op, operands = p
        kinds, args = '', []
        for x in operands:
            reg = YMM.match(x)
            kinds += 'r' if reg else 'o'
            args.append(reg.group(1) if reg else x)
        self.signatures.add(kinds)
        return f'QZ{kinds}({",".join([self.alias.get(op, op)] + args)})'

    def definitions(self) -> list[str]:
        lines = ['#define QZs(x) QZt(x)', '#define QZt(x) #x', f'#define QZc {CLOBBERS}']
        lines += [f'#define {a} {op}' for op, a in self.alias.items()]
        for kinds in sorted(self.signatures):
            params = ','.join(f'a{i}' for i in range(len(kinds)))
            parts = [('"%%ymm" #' if k == 'r' else '#') + f'a{i}' for i, k in enumerate(kinds)]
            body = ' ","'.join(parts)
            lines.append(f'#define QZ{kinds}(op,{params}) QZs(op) " " {body} "\\n"')
        return lines

    def names(self) -> list[str]:
        return ['QZs', 'QZt', 'QZc'] + list(self.alias.values()) + [f'QZ{k}' for k in sorted(self.signatures)]


def tokens(text: str):
    for m in TOKEN.finditer(text):
        yield m.lastgroup, m.group()


def compact(text: str) -> str:
    # Preprocessor directives stay on their own lines; everything else joins into long lines.
    logical = text.replace('\\\n', '')
    strings = [v for k, v in tokens(logical) if k == 'str']
    asm = Asm(strings)
    out_lines, line, last, at_line_start, directive = [], [], '', True, False

    def flush():
        nonlocal line, last
        if line:
            out_lines.append(''.join(line))
        line, last = [], ''

    pending_space = False
    for kind, value in tokens(logical):
        if kind == 'nl':
            if directive:
                flush()
                directive = False
            at_line_start = True
            pending_space = True
            continue
        if kind in ('ws', 'line', 'block'):
            pending_space = True
            continue
        if at_line_start and value == '#':
            flush()
            directive = True
        at_line_start = False
        if kind == 'str' and not directive:
            value = asm.rewrite(value)
            pending_space = pending_space or value[0] != '"'
        if pending_space and last and (
                (WORD.match(last[-1]) and WORD.match(value[0]))
                or (last[-1] in OPERATORS and value[0] in OPERATORS)
                or (last[-1] in 'eEpP' and value[0] in '+-')  # 0xE + 1 is not the pp-number 0xE+1
                or (last[-1] in '"\'' and WORD.match(value[0]))  # "s" X is not a literal with suffix X
                or directive):
            line.append(' ')
        pending_space = False
        line.append(value)
        last = value
        if not directive and value in (';', '}') and sum(map(len, line)) > 4000:
            flush()
    flush()
    lines = [x.replace(CLOBBERS.replace(', ', ','), 'QZc') for x in out_lines]
    uses = [i for i, x in enumerate(lines) if re.search(r'\bQZ', x)]
    if uses:
        first, last_use = uses[0], uses[-1]
        lines = lines[:first] + asm.definitions() + lines[first:last_use + 1] + \
            [f'#undef {n}' for n in asm.names()] + lines[last_use + 1:]
    return shorten('\n'.join(lines) + '\n')


def short_names(taken: set[str]):
    """Identifiers of growing length that are not in taken."""
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for c in letters:
        if c not in taken:
            yield c
    for c in letters:
        for d in letters + '0123456789':
            if c + d not in taken:
                yield c + d


def shorten(text: str) -> str:
    """Renames the QZ macros to the shortest identifiers the file does not use, most used first."""
    names = re.findall(r'\b[A-Za-z_$][\w$]*', text)
    counts = collections.Counter(n for n in names if n.startswith('QZ'))
    fresh = short_names({n for n in names if not n.startswith('QZ')})
    mapping = {name: next(fresh) for name, _ in counts.most_common()}
    return re.sub(r'\bQZ\w*', lambda m: mapping[m.group()], text)
