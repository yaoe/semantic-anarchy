/**
 * A tiny arithmetic evaluator for the numeric sidebar fields, so a box can be
 * fed `1024+256` (or `1024*1.5`, `(512+256)*2`) and turned into a plain number
 * on Enter/blur. Hand-rolled recursive descent — no `eval`, no dependency.
 *
 * Supports + - * / % ^, unary minus, parentheses and decimals. Anything else
 * (letters, stray symbols, division by zero, unbalanced parens) returns null,
 * which the caller treats as "leave what the user typed alone".
 */

export function evalExpr(src: string): number | null {
  const s = src.trim()
  if (!s || !/^[\d\s.+\-*/%^()]+$/.test(s)) return null

  let i = 0
  const skip = () => {
    while (i < s.length && s[i] === ' ') i++
  }
  const eat = (tok: string) => {
    skip()
    if (s.startsWith(tok, i)) {
      i += tok.length
      return true
    }
    return false
  }

  // primary := number | '(' expr ')' | ('-'|'+') primary
  const primary = (): number | null => {
    skip()
    if (eat('-')) {
      const v = primary()
      return v === null ? null : -v
    }
    if (eat('+')) return primary()
    if (eat('(')) {
      const v = sum()
      if (v === null || !eat(')')) return null
      return v
    }
    const m = /^\d*\.?\d+/.exec(s.slice(i))
    if (!m) return null
    i += m[0].length
    return parseFloat(m[0])
  }

  // power := primary ('^' power)?   -- right associative
  const power = (): number | null => {
    const base = primary()
    if (base === null) return null
    if (eat('^')) {
      const exp = power()
      return exp === null ? null : base ** exp
    }
    return base
  }

  // product := power (('*'|'/'|'%') power)*
  const product = (): number | null => {
    let acc = power()
    if (acc === null) return null
    for (;;) {
      skip()
      const op = s[i]
      if (op !== '*' && op !== '/' && op !== '%') return acc
      i++
      const rhs = power()
      if (rhs === null) return null
      if ((op === '/' || op === '%') && rhs === 0) return null
      acc = op === '*' ? acc * rhs : op === '/' ? acc / rhs : acc % rhs
    }
  }

  // sum := product (('+'|'-') product)*
  const sum = (): number | null => {
    let acc = product()
    if (acc === null) return null
    for (;;) {
      skip()
      const op = s[i]
      if (op !== '+' && op !== '-') return acc
      i++
      const rhs = product()
      if (rhs === null) return null
      acc = op === '+' ? acc + rhs : acc - rhs
    }
  }

  const out = sum()
  skip()
  if (out === null || i !== s.length || !Number.isFinite(out)) return null
  return out
}

/**
 * Evaluate `src` and render it back as a field value. `integer` rounds (width
 * and height must be whole pixels); otherwise a few decimals are kept. Returns
 * null when there is nothing to change — not an expression, or already the
 * number it evaluates to.
 */
export function evalField(src: string, integer: boolean): string | null {
  const v = evalExpr(src)
  if (v === null) return null
  const out = integer ? String(Math.round(v)) : String(Math.round(v * 1e6) / 1e6)
  return out === src.trim() ? null : out
}
