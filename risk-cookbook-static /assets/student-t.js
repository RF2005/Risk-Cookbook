import {
  parseNumberList,
  dot,
  matrixVector,
  buildCovarianceFromVols,
  cholesky,
  lowerTriangularVectorMultiply,
  randn,
  randomChiSquare,
  quantile,
  expectedShortfall,
  fmt,
  renderHistogram,
  renderLineChart,
  renderMultiLineChart
} from './utils.js'

function validate(inputs) {
  const errors = []
  const assetCount = Number(inputs.assetCount.value)
  const weights = parseNumberList(inputs.weights.value)
  const means = parseNumberList(inputs.means.value)
  const vols = parseNumberList(inputs.vols.value)
  const corr = Number(inputs.corr.value)
  const alpha = Number(inputs.alpha.value)
  const simulations = Number(inputs.simulations.value)
  const df = Number(inputs.df.value)

  if (!Number.isInteger(assetCount) || assetCount <= 0) errors.push('Number of assets must be a positive integer.')
  if (weights.length !== assetCount) errors.push('Weights must contain one value per asset.')
  if (means.length !== assetCount) errors.push('Mean returns must contain one value per asset.')
  if (vols.length !== assetCount) errors.push('Volatilities must contain one value per asset.')
  if (weights.some(Number.isNaN)) errors.push('Weights must be numeric.')
  if (means.some(Number.isNaN)) errors.push('Mean returns must be numeric.')
  if (vols.some(x => Number.isNaN(x) || x < 0)) errors.push('Volatilities must be numeric and nonnegative.')
  if (Number.isNaN(corr) || corr <= -1 || corr >= 1) errors.push('Common correlation must be between -1 and 1.')
  if (Number.isNaN(alpha) || alpha <= 0 || alpha >= 1) errors.push('Confidence level must be between 0 and 1.')
  if (!Number.isInteger(simulations) || simulations <= 0) errors.push('Simulations must be a positive integer.')
  if (Number.isNaN(df) || df <= 2) errors.push('Degrees of freedom must be greater than 2.')

  return {
    errors,
    parsed: errors.length ? null : { assetCount, weights, means, vols, corr, alpha, simulations, df }
  }
}

function buildPython(raw) {
  const weights = raw.weights.value.trim() || '...'
  const means = raw.means.value.trim() || '...'
  const vols = raw.vols.value.trim() || '...'
  const corr = raw.corr.value.trim() || '...'
  const df = raw.df.value.trim() || '...'
  const alpha = raw.alpha.value.trim() || '...'
  const simulations = raw.simulations.value.trim() || '...'

  return `import numpy as np\n\nweights = np.array([${weights}])\nmeans = np.array([${means}])\nvols = np.array([${vols}])\ncorr = ${corr}\ndf = ${df}\nalpha = ${alpha}\nsimulations = ${simulations}\n\nn = len(weights)\ncorr_matrix = np.full((n, n), corr)\nnp.fill_diagonal(corr_matrix, 1.0)\ncov = np.outer(vols, vols) * corr_matrix\n\nz = np.random.multivariate_normal(np.zeros(n), cov, size=simulations)\nchi = np.random.chisquare(df, size=simulations)\nreturns = means + z * np.sqrt(df / chi)[:, None]\nlosses = -(returns @ weights)\n\nvar_loss = np.quantile(losses, alpha)\nes_loss = losses[losses >= var_loss].mean()\n\npath_steps = 30\npath_count = 12`
}

function simulateGaussianLosses(parsed, covariance) {
  const chol = cholesky(covariance)
  const losses = []
  for (let s = 0; s < parsed.simulations; s++) {
    const z = Array.from({ length: parsed.assetCount }, () => randn())
    const correlated = lowerTriangularVectorMultiply(chol, z)
    const returns = correlated.map((x, i) => parsed.means[i] + x)
    losses.push(-dot(parsed.weights, returns))
  }
  return losses
}

function simulateTPath(chol, means, weights, df, steps) {
  let value = 1
  const points = [{ x: 0, y: value }]
  for (let step = 1; step <= steps; step++) {
    const z = Array.from({ length: weights.length }, () => randn())
    const correlated = lowerTriangularVectorMultiply(chol, z)
    const chi = randomChiSquare(df)
    const scale = Math.sqrt(df / chi)
    const returns = correlated.map((x, i) => means[i] + x * scale)
    const portfolioReturn = dot(weights, returns)
    value *= 1 + portfolioReturn
    points.push({ x: step, y: value })
  }
  return points
}

function runModel(parsed) {
  const covariance = buildCovarianceFromVols(parsed.vols, parsed.corr)
  const chol = cholesky(covariance)
  const tLosses = []
  for (let s = 0; s < parsed.simulations; s++) {
    const z = Array.from({ length: parsed.assetCount }, () => randn())
    const correlated = lowerTriangularVectorMultiply(chol, z)
    const chi = randomChiSquare(parsed.df)
    const scale = Math.sqrt(parsed.df / chi)
    const returns = correlated.map((x, i) => parsed.means[i] + x * scale)
    tLosses.push(-dot(parsed.weights, returns))
  }
  const gaussianLosses = simulateGaussianLosses(parsed, covariance)
  const tVar = quantile(tLosses, parsed.alpha)
  const gVar = quantile(gaussianLosses, parsed.alpha)
  const paths = Array.from({ length: 12 }, (_, i) => ({
    name: `Path ${i + 1}`,
    points: simulateTPath(chol, parsed.means, parsed.weights, parsed.df, 30)
  }))
  return {
    meanLoss: -dot(parsed.weights, parsed.means),
    stdevLoss: Math.sqrt(dot(parsed.weights, matrixVector(covariance, parsed.weights))),
    varLoss: tVar,
    esLoss: expectedShortfall(tLosses, tVar),
    gaussianVar: gVar,
    gaussianEs: expectedShortfall(gaussianLosses, gVar),
    tLosses,
    paths
  }
}

function renderResults(result, parsed) {
  document.getElementById('student-results').innerHTML = `
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Expected loss</div><div class="metric-value">${fmt(result.meanLoss)}</div></div>
      <div class="metric"><div class="metric-label">Loss volatility</div><div class="metric-value">${fmt(result.stdevLoss)}</div></div>
      <div class="metric"><div class="metric-label">Student-t VaR</div><div class="metric-value">${fmt(result.varLoss)}</div></div>
      <div class="metric"><div class="metric-label">Student-t ES</div><div class="metric-value">${fmt(result.esLoss)}</div></div>
      <div class="metric"><div class="metric-label">Gaussian VaR</div><div class="metric-value">${fmt(result.gaussianVar)}</div></div>
      <div class="metric"><div class="metric-label">Gaussian ES</div><div class="metric-value">${fmt(result.gaussianEs)}</div></div>
      <div class="metric"><div class="metric-label">VaR difference</div><div class="metric-value">${fmt(result.varLoss - result.gaussianVar)}</div></div>
      <div class="metric"><div class="metric-label">ES difference</div><div class="metric-value">${fmt(result.esLoss - result.gaussianEs)}</div></div>
    </div>
    <p class="small">The histogram shows one-horizon tail risk. The path view below uses the same Student-t assumptions to show how rare large shocks can distort the path of portfolio value.</p>`

  renderHistogram(document.getElementById('student-histogram'), result.tLosses, [
    { value: result.varLoss, label: 't VaR', color: '#1d6b2f' },
    { value: result.esLoss, label: 't ES', color: '#a33a00' }
  ], 'Student-t loss distribution')

  renderMultiLineChart(document.getElementById('student-paths'), result.paths, 'Illustrative Student-t portfolio paths')

  const dfPoints = [3, 4, 6, 10, 20, 50].map(df => {
    const one = runModel({ ...parsed, df, simulations: Math.min(parsed.simulations, 3000) })
    return { x: df, y: one.esLoss }
  })
  renderLineChart(document.getElementById('student-sensitivity'), dfPoints, 'Expected shortfall vs degrees of freedom')
}

function updatePython(inputs) {
  document.getElementById('student-python').textContent = buildPython(inputs)
}

function hasAnyValue(inputs) {
  return Object.values(inputs).some(input => input.value.trim() !== '')
}

function updateCanRun(inputs, showErrors = false) {
  const validation = validate(inputs)
  const anyValue = hasAnyValue(inputs)
  const canRun = Object.values(inputs).every(input => input.value.trim() !== '') && validation.errors.length === 0
  document.getElementById('student-run').disabled = !canRun
  document.getElementById('student-errors').innerHTML = showErrors && anyValue ? validation.errors.map(error => `<div class="notice">${error}</div>`).join('') : ''
  updatePython(inputs)
  return validation
}

function clearOutputs() {
  document.getElementById('student-results').innerHTML = '<div class="empty">Enter inputs and run the model to generate results.</div>'
  document.getElementById('student-histogram').innerHTML = ''
  document.getElementById('student-paths').innerHTML = ''
  document.getElementById('student-sensitivity').innerHTML = ''
}

export function setupStudentTPage() {
  const ids = ['assetCount', 'weights', 'means', 'vols', 'corr', 'df', 'alpha', 'simulations']
  const inputs = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]))
  ids.forEach(id => inputs[id].addEventListener('input', () => updateCanRun(inputs, hasAnyValue(inputs))))
  document.getElementById('student-run').addEventListener('click', () => {
    const validation = updateCanRun(inputs, true)
    if (validation.errors.length) return
    const result = runModel(validation.parsed)
    renderResults(result, validation.parsed)
    document.getElementById('student-errors').innerHTML = '<div class="success">Model ran successfully.</div>'
  })
  document.getElementById('student-clear').addEventListener('click', () => {
    ids.forEach(id => { inputs[id].value = '' })
    clearOutputs()
    updateCanRun(inputs, false)
  })
  clearOutputs()
  updateCanRun(inputs, false)
}
