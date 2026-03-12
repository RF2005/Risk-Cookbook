import {
  parseNumberList,
  dot,
  quantile,
  expectedShortfall,
  mean,
  stdev,
  fmt,
  renderHistogram,
  renderLineChart,
  renderMultiLineChart,
  csvToRows
} from './utils.js'

function parseCsv(raw) {
  const rows = csvToRows(raw)
  if (rows.length < 2) throw new Error('CSV must include a header and at least one data row.')
  const header = rows[0]
  const dataRows = rows.slice(1)
  return {
    header,
    rows: dataRows.map(row => ({ date: row[0], values: row.slice(1).map(Number) }))
  }
}

function validate(inputs) {
  const errors = []
  const assetCount = Number(inputs.assetCount.value)
  const weights = parseNumberList(inputs.weights.value)
  const alpha = Number(inputs.alpha.value)
  const lookback = Number(inputs.lookback.value)
  const csv = inputs.csv.value.trim()

  if (!Number.isInteger(assetCount) || assetCount <= 0) errors.push('Number of assets must be a positive integer.')
  if (weights.length !== assetCount) errors.push('Weights must contain one value per asset.')
  if (weights.some(Number.isNaN)) errors.push('Weights must be numeric.')
  if (Number.isNaN(alpha) || alpha <= 0 || alpha >= 1) errors.push('Confidence level must be between 0 and 1.')
  if (!Number.isInteger(lookback) || lookback <= 0) errors.push('Lookback window must be a positive integer.')
  if (!csv) errors.push('Paste return data in CSV format.')

  let parsedCsv = null
  if (!errors.length) {
    try {
      parsedCsv = parseCsv(csv)
      if (parsedCsv.header.length - 1 !== assetCount) errors.push('CSV must contain one return column per asset.')
      if (parsedCsv.rows.some(row => row.values.length !== assetCount || row.values.some(Number.isNaN))) {
        errors.push('All return rows must contain numeric values for every asset.')
      }
      if (parsedCsv.rows.length < lookback) errors.push('Lookback window exceeds available rows.')
    } catch (error) {
      errors.push(error.message)
    }
  }

  return {
    errors,
    parsed: errors.length ? null : { assetCount, weights, alpha, lookback, parsedCsv }
  }
}

function buildPython(raw) {
  const weights = raw.weights.value.trim() || '...'
  const alpha = raw.alpha.value.trim() || '...'
  const lookback = raw.lookback.value.trim() || '...'

  return `import pandas as pd\nimport numpy as np\n\nweights = np.array([${weights}])\nalpha = ${alpha}\nlookback = ${lookback}\n\ndf = pd.read_csv('returns.csv')\nreturns = df.iloc[-lookback:, 1:].to_numpy()\nlosses = -(returns @ weights)\n\nvar_loss = np.quantile(losses, alpha)\nes_loss = losses[losses >= var_loss].mean()\nportfolio_returns = returns @ weights\nportfolio_value = np.cumprod(1 + portfolio_returns)`
}

function runModel(parsed) {
  const windowRows = parsed.parsedCsv.rows.slice(-parsed.lookback)
  const scenarios = windowRows.map(row => ({
    date: row.date,
    loss: -dot(parsed.weights, row.values),
    values: row.values,
    portfolioReturn: dot(parsed.weights, row.values)
  }))
  const losses = scenarios.map(row => row.loss)
  const varLoss = quantile(losses, parsed.alpha)
  const esLoss = expectedShortfall(losses, varLoss)
  const worst = [...scenarios].sort((a, b) => b.loss - a.loss).slice(0, 5)

  let value = 1
  const portfolioPath = scenarios.map((row, index) => {
    value *= 1 + row.portfolioReturn
    return { x: index + 1, y: value }
  })

  return {
    meanLoss: mean(losses),
    stdevLoss: stdev(losses),
    varLoss,
    esLoss,
    observations: losses.length,
    tailCount: losses.filter(x => x >= varLoss).length,
    losses,
    worst,
    portfolioPath
  }
}

function renderWorstTable(rows) {
  return `<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Date</th><th>Loss</th></tr></thead><tbody>${rows.map((row, i) => `<tr><td>${i + 1}</td><td>${row.date}</td><td>${fmt(row.loss)}</td></tr>`).join('')}</tbody></table></div>`
}

function renderResults(result, parsed) {
  document.getElementById('historical-results').innerHTML = `
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Expected loss</div><div class="metric-value">${fmt(result.meanLoss)}</div></div>
      <div class="metric"><div class="metric-label">Loss volatility</div><div class="metric-value">${fmt(result.stdevLoss)}</div></div>
      <div class="metric"><div class="metric-label">Historical VaR</div><div class="metric-value">${fmt(result.varLoss)}</div></div>
      <div class="metric"><div class="metric-label">Historical ES</div><div class="metric-value">${fmt(result.esLoss)}</div></div>
      <div class="metric"><div class="metric-label">Observations</div><div class="metric-value">${result.observations}</div></div>
      <div class="metric"><div class="metric-label">Tail observations</div><div class="metric-value">${result.tailCount}</div></div>
    </div>
    <p class="small">This page is a scenario replay rather than a parametric Monte Carlo. The line chart below shows the cumulative portfolio path implied by the pasted return history.</p>`

  renderHistogram(document.getElementById('historical-histogram'), result.losses, [
    { value: result.varLoss, label: 'VaR', color: '#1d6b2f' },
    { value: result.esLoss, label: 'ES', color: '#a33a00' }
  ], 'Historical loss distribution')

  renderMultiLineChart(document.getElementById('historical-paths'), [{ name: 'Portfolio value', points: result.portfolioPath }], 'Cumulative portfolio path over the selected history')

  const windows = [Math.max(25, Math.floor(parsed.lookback / 4)), Math.max(50, Math.floor(parsed.lookback / 2)), parsed.lookback]
  const uniqueWindows = [...new Set(windows)].filter(x => x <= parsed.parsedCsv.rows.length)
  const points = uniqueWindows.map(window => {
    const mini = runModel({ ...parsed, lookback: window })
    return { x: window, y: mini.varLoss }
  })
  renderLineChart(document.getElementById('historical-sensitivity'), points, 'Historical VaR vs lookback window')
  document.getElementById('historical-worst').innerHTML = renderWorstTable(result.worst)
}

function updatePython(inputs) {
  document.getElementById('historical-python').textContent = buildPython(inputs)
}

function hasAnyValue(inputs) {
  return Object.values(inputs).some(input => input.value.trim() !== '')
}

function updateCanRun(inputs, showErrors = false) {
  const validation = validate(inputs)
  const anyValue = hasAnyValue(inputs)
  const canRun = Object.values(inputs).every(input => input.value.trim() !== '') && validation.errors.length === 0
  document.getElementById('historical-run').disabled = !canRun
  document.getElementById('historical-errors').innerHTML = showErrors && anyValue ? validation.errors.map(error => `<div class="notice">${error}</div>`).join('') : ''
  updatePython(inputs)
  return validation
}

function clearOutputs() {
  document.getElementById('historical-results').innerHTML = '<div class="empty">Enter inputs and run the model to generate results.</div>'
  document.getElementById('historical-histogram').innerHTML = ''
  document.getElementById('historical-paths').innerHTML = ''
  document.getElementById('historical-sensitivity').innerHTML = ''
  document.getElementById('historical-worst').innerHTML = ''
}

export function setupHistoricalPage() {
  const ids = ['assetCount', 'weights', 'alpha', 'lookback', 'csv']
  const inputs = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]))
  ids.forEach(id => inputs[id].addEventListener('input', () => updateCanRun(inputs, hasAnyValue(inputs))))
  document.getElementById('historical-run').addEventListener('click', () => {
    const validation = updateCanRun(inputs, true)
    if (validation.errors.length) return
    const result = runModel(validation.parsed)
    renderResults(result, validation.parsed)
    document.getElementById('historical-errors').innerHTML = '<div class="success">Model ran successfully.</div>'
  })
  document.getElementById('historical-clear').addEventListener('click', () => {
    ids.forEach(id => { inputs[id].value = '' })
    clearOutputs()
    updateCanRun(inputs, false)
  })
  clearOutputs()
  updateCanRun(inputs, false)
}
