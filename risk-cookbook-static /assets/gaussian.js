import {
  parseNumberList,
  dot,
  matrixVector,
  buildCovarianceFromVols,
  cholesky,
  lowerTriangularVectorMultiply,
  randn,
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

  return {
    errors,
    parsed: errors.length ? null : { assetCount, weights, means, vols, corr, alpha, simulations }
  }
}

function buildPython(raw) {
  const weights = raw.weights.value.trim() || '...'
  const means = raw.means.value.trim() || '...'
  const vols = raw.vols.value.trim() || '...'
  const corr = raw.corr.value.trim() || '...'
  const alpha = raw.alpha.value.trim() || '...'
  const simulations = raw.simulations.value.trim() || '...'

  return `import numpy as np\n\nweights = np.array([${weights}])\nmeans = np.array([${means}])\nvols = np.array([${vols}])\ncorr = ${corr}\nalpha = ${alpha}\nsimulations = ${simulations}\n\nn = len(weights)\ncorr_matrix = np.full((n, n), corr)\nnp.fill_diagonal(corr_matrix, 1.0)\ncov = np.outer(vols, vols) * corr_matrix\n\nreturns = np.random.multivariate_normal(means, cov, size=simulations)\nlosses = -(returns @ weights)\n\nmean_loss = -(weights @ means)\nstdev_loss = np.sqrt(weights @ cov @ weights)\nvar_loss = np.quantile(losses, alpha)\nes_loss = losses[losses >= var_loss].mean()\n\npath_steps = 30\npath_count = 12\npath_returns = np.random.multivariate_normal(means, cov, size=(path_steps, path_count))\nportfolio_returns = path_returns @ weights\nportfolio_values = np.vstack([np.ones(path_count), np.cumprod(1 + portfolio_returns, axis=0)])`
}

function simulatePortfolioPath(chol, means, weights, steps) {
  let value = 1
  const points = [{ x: 0, y: value }]
  for (let step = 1; step <= steps; step++) {
    const z = Array.from({ length: weights.length }, () => randn())
    const correlated = lowerTriangularVectorMultiply(chol, z)
    const returns = correlated.map((x, i) => means[i] + x)
    const portfolioReturn = dot(weights, returns)
    value *= 1 + portfolioReturn
    points.push({ x: step, y: value })
  }
  return points
}

function runModel(parsed) {
  const covariance = buildCovarianceFromVols(parsed.vols, parsed.corr)
  const chol = cholesky(covariance)
  const losses = []
  for (let s = 0; s < parsed.simulations; s++) {
    const z = Array.from({ length: parsed.assetCount }, () => randn())
    const correlated = lowerTriangularVectorMultiply(chol, z)
    const returns = correlated.map((x, i) => parsed.means[i] + x)
    losses.push(-dot(parsed.weights, returns))
  }
  const meanLoss = -dot(parsed.weights, parsed.means)
  const stdevLoss = Math.sqrt(dot(parsed.weights, matrixVector(covariance, parsed.weights)))
  const varLoss = quantile(losses, parsed.alpha)
  const esLoss = expectedShortfall(losses, varLoss)
  const paths = Array.from({ length: 12 }, (_, i) => ({
    name: `Path ${i + 1}`,
    points: simulatePortfolioPath(chol, parsed.means, parsed.weights, 30)
  }))
  return { meanLoss, stdevLoss, varLoss, esLoss, losses, paths }
}

function renderResults(result, parsed) {
  document.getElementById('gaussian-results').innerHTML = `
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Expected loss</div><div class="metric-value">${fmt(result.meanLoss)}</div></div>
      <div class="metric"><div class="metric-label">Loss volatility</div><div class="metric-value">${fmt(result.stdevLoss)}</div></div>
      <div class="metric"><div class="metric-label">VaR</div><div class="metric-value">${fmt(result.varLoss)}</div></div>
      <div class="metric"><div class="metric-label">Expected shortfall</div><div class="metric-value">${fmt(result.esLoss)}</div></div>
    </div>
    <p class="small">The histogram shows one-horizon losses. The path view below shows 30-step portfolio-value paths generated from the same Gaussian assumptions.</p>`

  renderHistogram(document.getElementById('gaussian-histogram'), result.losses, [
    { value: result.varLoss, label: 'VaR', color: '#1d6b2f' },
    { value: result.esLoss, label: 'ES', color: '#a33a00' }
  ])

  renderMultiLineChart(document.getElementById('gaussian-paths'), result.paths, 'Illustrative Gaussian portfolio paths')

  const corrPoints = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8].map(rho => {
    const covariance = buildCovarianceFromVols(parsed.vols, rho)
    const sigma = Math.sqrt(dot(parsed.weights, matrixVector(covariance, parsed.weights)))
    return { x: rho, y: sigma }
  })
  renderLineChart(document.getElementById('gaussian-sensitivity'), corrPoints, 'Loss volatility vs correlation')
}

function updatePython(inputs) {
  document.getElementById('gaussian-python').textContent = buildPython(inputs)
}

function hasAnyValue(inputs) {
  return Object.values(inputs).some(input => input.value.trim() !== '')
}

function updateCanRun(inputs, showErrors = false) {
  const validation = validate(inputs)
  const anyValue = hasAnyValue(inputs)
  const canRun = Object.values(inputs).every(input => input.value.trim() !== '') && validation.errors.length === 0
  document.getElementById('gaussian-run').disabled = !canRun
  const errorsEl = document.getElementById('gaussian-errors')
  errorsEl.innerHTML = showErrors && anyValue ? validation.errors.map(error => `<div class="notice">${error}</div>`).join('') : ''
  updatePython(inputs)
  return validation
}

function clearOutputs() {
  document.getElementById('gaussian-results').innerHTML = '<div class="empty">Enter inputs and run the model to generate results.</div>'
  document.getElementById('gaussian-histogram').innerHTML = ''
  document.getElementById('gaussian-paths').innerHTML = ''
  document.getElementById('gaussian-sensitivity').innerHTML = ''
}

export function setupGaussianPage() {
  const ids = ['assetCount', 'weights', 'means', 'vols', 'corr', 'alpha', 'simulations']
  const inputs = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]))
  ids.forEach(id => inputs[id].addEventListener('input', () => updateCanRun(inputs, hasAnyValue(inputs))))
  document.getElementById('gaussian-run').addEventListener('click', () => {
    const validation = updateCanRun(inputs, true)
    if (validation.errors.length) return
    const result = runModel(validation.parsed)
    renderResults(result, validation.parsed)
    document.getElementById('gaussian-errors').innerHTML = '<div class="success">Model ran successfully.</div>'
  })
  document.getElementById('gaussian-clear').addEventListener('click', () => {
    ids.forEach(id => { inputs[id].value = '' })
    clearOutputs()
    updateCanRun(inputs, false)
  })
  clearOutputs()
  updateCanRun(inputs, false)
}
