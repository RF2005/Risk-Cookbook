export function parseNumberList(raw) {
  return raw
    .split(',')
    .map(x => x.trim())
    .filter(Boolean)
    .map(Number)
}

export function parseMatrix(raw) {
  return raw
    .trim()
    .split(/\n+/)
    .map(line => line.split(',').map(x => Number(x.trim())))
    .filter(row => row.length && row.some(x => !Number.isNaN(x)))
}

export function dot(a, b) {
  return a.reduce((sum, x, i) => sum + x * b[i], 0)
}

export function mean(values) {
  return values.reduce((sum, x) => sum + x, 0) / values.length
}

export function stdev(values) {
  const m = mean(values)
  return Math.sqrt(values.reduce((sum, x) => sum + (x - m) ** 2, 0) / values.length)
}

export function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))
  return sorted[index]
}

export function expectedShortfall(values, varLoss) {
  const tail = values.filter(x => x >= varLoss)
  return mean(tail)
}

export function buildCommonCorrelationMatrix(n, corr) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : corr))
  )
}

export function buildCovarianceFromVols(vols, corr) {
  const rho = buildCommonCorrelationMatrix(vols.length, corr)
  return rho.map((row, i) => row.map((v, j) => v * vols[i] * vols[j]))
}

export function matrixVector(matrix, vector) {
  return matrix.map(row => dot(row, vector))
}

export function cholesky(matrix) {
  const n = matrix.length
  const l = Array.from({ length: n }, () => Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) sum += l[i][k] * l[j][k]
      if (i === j) {
        const value = matrix[i][i] - sum
        if (value <= 0) throw new Error('Matrix must be positive definite.')
        l[i][j] = Math.sqrt(value)
      } else {
        l[i][j] = (matrix[i][j] - sum) / l[j][j]
      }
    }
  }
  return l
}

export function lowerTriangularVectorMultiply(matrix, vector) {
  return matrix.map((row, i) => {
    let sum = 0
    for (let j = 0; j <= i; j++) sum += row[j] * vector[j]
    return sum
  })
}

export function randn() {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export function randomGamma(shape) {
  if (shape < 1) {
    const u = Math.random()
    return randomGamma(1 + shape) * Math.pow(u, 1 / shape)
  }
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  while (true) {
    const x = randn()
    const v = Math.pow(1 + c * x, 3)
    if (v <= 0) continue
    const u = Math.random()
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return d * v
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

export function randomChiSquare(df) {
  return 2 * randomGamma(df / 2)
}

export function fmt(value) {
  if (!Number.isFinite(value)) return '—'
  return Number(value).toFixed(6)
}

export function histogram(values, binCount = 30) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const width = max === min ? 1 : (max - min) / binCount
  const bins = Array.from({ length: binCount }, (_, i) => ({
    x0: min + i * width,
    x1: i === binCount - 1 ? max : min + (i + 1) * width,
    count: 0
  }))
  values.forEach(value => {
    const index = max === min ? 0 : Math.min(binCount - 1, Math.floor((value - min) / width))
    bins[index].count += 1
  })
  return bins
}

export function renderHistogram(container, values, markers = [], title = 'Loss distribution') {
  container.innerHTML = ''
  if (!values.length) return
  const bins = histogram(values, 28)
  const w = 640
  const h = 280
  const m = { t: 10, r: 16, b: 34, l: 44 }
  const maxCount = Math.max(...bins.map(b => b.count), 1)
  const minX = bins[0].x0
  const maxX = bins[bins.length - 1].x1
  const xScale = x => m.l + ((x - minX) / (maxX - minX || 1)) * (w - m.l - m.r)
  const yScale = y => h - m.b - (y / maxCount) * (h - m.t - m.b)
  const barWidth = (w - m.l - m.r) / bins.length
  const rects = bins.map((bin, i) => {
    const x = m.l + i * barWidth
    const y = yScale(bin.count)
    const height = h - m.b - y
    return `<rect x="${x}" y="${y}" width="${Math.max(barWidth - 2, 1)}" height="${height}" fill="#111" opacity="0.82" />`
  }).join('')
  const lines = markers.map(marker => {
    const x = xScale(marker.value)
    return `<line x1="${x}" y1="${m.t}" x2="${x}" y2="${h - m.b}" stroke="${marker.color}" stroke-width="2" />
      <text x="${Math.min(x + 6, w - 80)}" y="${m.t + 14}" class="chart-label">${marker.label}</text>`
  }).join('')
  container.innerHTML = `
    <div class="plot-shell">
      <div class="plot-title">${title}</div>
      <svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}">
        <line x1="${m.l}" y1="${h - m.b}" x2="${w - m.r}" y2="${h - m.b}" stroke="#999" />
        <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${h - m.b}" stroke="#999" />
        ${rects}
        ${lines}
        <text x="${m.l}" y="${h - 10}" class="chart-label">${minX.toFixed(4)}</text>
        <text x="${w - m.r - 40}" y="${h - 10}" class="chart-label">${maxX.toFixed(4)}</text>
      </svg>
    </div>`
}

export function renderLineChart(container, points, title, xLabelPrefix = '') {
  container.innerHTML = ''
  if (!points.length) return
  const w = 640
  const h = 280
  const m = { t: 10, r: 16, b: 34, l: 44 }
  const minX = Math.min(...points.map(p => p.x))
  const maxX = Math.max(...points.map(p => p.x))
  const minY = Math.min(...points.map(p => p.y))
  const maxY = Math.max(...points.map(p => p.y))
  const xScale = x => m.l + ((x - minX) / (maxX - minX || 1)) * (w - m.l - m.r)
  const yScale = y => h - m.b - ((y - minY) / (maxY - minY || 1)) * (h - m.t - m.b)
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x)} ${yScale(p.y)}`).join(' ')
  const dots = points.map(p => `<circle cx="${xScale(p.x)}" cy="${yScale(p.y)}" r="3" fill="#111" />`).join('')
  const labels = points.map(p => `<text x="${xScale(p.x) - 8}" y="${h - 10}" class="chart-label">${xLabelPrefix}${p.x}</text>`).join('')
  container.innerHTML = `
    <div class="plot-shell">
      <div class="plot-title">${title}</div>
      <svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}">
        <line x1="${m.l}" y1="${h - m.b}" x2="${w - m.r}" y2="${h - m.b}" stroke="#999" />
        <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${h - m.b}" stroke="#999" />
        <path d="${path}" fill="none" stroke="#111" stroke-width="2" />
        ${dots}
        ${labels}
        <text x="${m.l}" y="${m.t + 12}" class="chart-label">${maxY.toFixed(4)}</text>
        <text x="${m.l}" y="${h - m.b}" class="chart-label">${minY.toFixed(4)}</text>
      </svg>
    </div>`
}

export function renderMultiLineChart(container, series, title) {
  container.innerHTML = ''
  if (!series.length || !series.some(one => one.points.length)) return
  const w = 640
  const h = 280
  const m = { t: 10, r: 16, b: 34, l: 44 }
  const allPoints = series.flatMap(one => one.points)
  const minX = Math.min(...allPoints.map(p => p.x))
  const maxX = Math.max(...allPoints.map(p => p.x))
  const minY = Math.min(...allPoints.map(p => p.y))
  const maxY = Math.max(...allPoints.map(p => p.y))
  const xScale = x => m.l + ((x - minX) / (maxX - minX || 1)) * (w - m.l - m.r)
  const yScale = y => h - m.b - ((y - minY) / (maxY - minY || 1)) * (h - m.t - m.b)
  const palette = ['#111111', '#444444', '#777777', '#999999', '#222222', '#555555', '#888888']
  const lines = series.map((one, index) => {
    const color = one.color || palette[index % palette.length]
    const path = one.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x)} ${yScale(p.y)}`).join(' ')
    return `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8" opacity="0.92" />`
  }).join('')
  const legend = series.slice(0, 5).map((one, index) => {
    const color = one.color || palette[index % palette.length]
    return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${one.name || `Series ${index + 1}`}</span>`
  }).join('')
  container.innerHTML = `
    <div class="plot-shell">
      <div class="plot-title">${title}</div>
      <div class="legend-row">${legend}</div>
      <svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}">
        <line x1="${m.l}" y1="${h - m.b}" x2="${w - m.r}" y2="${h - m.b}" stroke="#999" />
        <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${h - m.b}" stroke="#999" />
        ${lines}
        <text x="${m.l}" y="${h - 10}" class="chart-label">${minX.toFixed(0)}</text>
        <text x="${w - m.r - 20}" y="${h - 10}" class="chart-label">${maxX.toFixed(0)}</text>
        <text x="${m.l}" y="${m.t + 12}" class="chart-label">${maxY.toFixed(4)}</text>
        <text x="${m.l}" y="${h - m.b}" class="chart-label">${minY.toFixed(4)}</text>
      </svg>
    </div>`
}

export function csvToRows(raw) {
  return raw.trim().split(/\n+/).map(line => line.split(',').map(x => x.trim()))
}
