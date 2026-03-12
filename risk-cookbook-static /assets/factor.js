import {
  parseNumberList,
  parseMatrix,
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
  const factorCount = Number(inputs.factorCount.value)
  const weights = parseNumberList(inputs.weights.value)
  const factorMeans = parseNumberList(inputs.factorMeans.value)
  const factorVols = parseNumberList(inputs.factorVols.value)
  const specificVols = parseNumberList(inputs.specificVols.value)
  const exposures = parseMatrix(inputs.exposures.value)
  const corr = Number(inputs.corr.value)
  const alpha = Number(inputs.alpha.value)
  const simulations = Number(inputs.simulations.value)

  if (!Number.isInteger(assetCount) || assetCount <= 0) errors.push('Number of assets must be a positive integer.')
  if (!Number.isInteger(factorCount) || factorCount <= 0) errors.push('Number of factors must be a positive integer.')
  if (weights.length !== assetCount) errors.push('Weights must contain one value per asset.')
  if (factorMeans.length !== factorCount) errors.push('Factor means must contain one value per factor.')
  if (factorVols.length !== factorCount) errors.push('Factor volatilities must contain one value per factor.')
  if (specificVols.length !== assetCount) errors.push('Specific volatilities must contain one value per asset.')
  if (exposures.length !== assetCount || exposures.some(row => row.length !== factorCount || row.some(Number.isNaN))) {
    errors.push('Exposure matrix must contain one row per asset and one column per factor.')
  }
  if (weights.some(Number.isNaN)) errors.push('Weights must be numeric.')
  if (factorMeans.some(Number.isNaN)) errors.push('Factor means must be numeric.')
  if (factorVols.some(x => Number.isNaN(x) || x < 0)) errors.push('Factor volatilities must be numeric and nonnegative.')
  if (specificVols.some(x => Number.isNaN(x) || x < 0)) errors.push('Specific volatilities must be numeric and nonnegative.')
  if (Number.isNaN(corr) || corr <= -1 || corr >= 1) errors.push('Common factor correlation must be between -1 and 1.')
  if (Number.isNaN(alpha) || alpha <= 0 || alpha >= 1) errors.push('Confidence level must be between 0 and 1.')
  if (!Number.isInteger(simulations) || simulations <= 0) errors.push('Simulations must be a positive integer.')

  return {
    errors,
    parsed: errors.length ? null : { assetCount, factorCount, weights, factorMeans, factorVols, specificVols, exposures, corr, alpha, simulations }
  }
}

function buildPython(raw) {
  const weights = raw.weights.value.trim() || '...'
  const factorMeans = raw.factorMeans.value.trim() || '...'
  const factorVols = raw.factorVols.value.trim() || '...'
  const specificVols = raw.specificVols.value.trim() || '...'
  const exposures = raw.exposures.value.trim()
    ? '[' + raw.exposures.value.trim().split(/\n+/).map(row => `[${row}]`).join(', ') + ']'
    : '[[...], [...]]'
  const corr = raw.corr.value.trim() || '...'
  const alpha = raw.alpha.value.trim() || '...'
  const simulations = raw.simulations.value.trim() || '...'

  return `import numpy as np\n\nweights = np.array([${weights}])\nB = np.array(${exposures})\nfactor_means = np.array([${factorMeans}])\nfactor_vols = np.array([${factorVols}])\nspecific_vols = np.array([${specificVols}])\ncorr = ${corr}\nalpha = ${alpha}\nsimulations = ${simulations}\n\nk = len(factor_means)\nrho = np.full((k, k), corr)\nnp.fill_diagonal(rho, 1.0)\nfactor_cov = np.outer(factor_vols, factor_vols) * rho\n\nfactors = np.random.multivariate_normal(factor_means, factor_cov, size=simulations)\nspecific = np.random.normal(0, specific_vols, size=(simulations, len(weights)))\nreturns = factors @ B.T + specific\nlosses = -(returns @ weights)\n\nvar_loss = np.quantile(losses, alpha)\nes_loss = losses[losses >= var_loss].mean()`
}

function buildAssetCovariance(parsed) {
  const factorCov = buildCovarianceFromVols(parsed.factorVols, parsed.corr)
  const assetCov = Array.from({ length: parsed.assetCount }, () => Array(parsed.assetCount).fill(0))
  for (let i = 0; i < parsed.assetCount; i++) {
    for (let j = 0; j < parsed.assetCount; j++) {
      let sum = 0
      for (let a = 0; a < parsed.factorCount; a++) {
        for (let b = 0; b < parsed.factorCount; b++) {
          sum += parsed.exposures[i][a] * factorCov[a][b] * parsed.exposures[j][b]
        }
      }
      if (i === j) sum += parsed.specificVols[i] ** 2
      assetCov[i][j] = sum
    }
  }
  return { factorCov, assetCov }
}

function simulateFactorPath(parsed, factorChol, steps) {
  let value = 1
  const points = [{ x: 0, y: value }]
  for (let step = 1; step <= steps; step++) {
    const zf = Array.from({ length: parsed.factorCount }, () => randn())
    const factorShock = lowerTriangularVectorMultiply(factorChol, zf).map((x, i) => x + parsed.factorMeans[i])
    const specificShock = parsed.specificVols.map(vol => randn() * vol)
    const assetReturns = parsed.exposures.map((row, i) => dot(row, factorShock) + specificShock[i])
    const portfolioReturn = dot(parsed.weights, assetReturns)
    value *= 1 + portfolioReturn
    points.push({ x: step, y: value })
  }
  return points
}

function runModel(parsed) {
  const { factorCov, assetCov } = buildAssetCovariance(parsed)
  const factorChol = cholesky(factorCov)
  const losses = []
  for (let s = 0; s < parsed.simulations; s++) {
    const zf = Array.from({ length: parsed.factorCount }, () => randn())
    const factorShock = lowerTriangularVectorMultiply(factorChol, zf).map((x, i) => x + parsed.factorMeans[i])
    const specificShock = parsed.specificVols.map(vol => randn() * vol)
    const assetReturns = parsed.exposures.map((row, i) => dot(row, factorShock) + specificShock[i])
    losses.push(-dot(parsed.weights, assetReturns))
  }
  const factorContribution = parsed.exposures[0].map((_, factorIndex) => {
    const loading = parsed.weights.reduce((sum, w, i) => sum + w * parsed.exposures[i][factorIndex], 0)
    return {
      x: factorIndex + 1,
      y: loading * loading * factorCov[factorIndex][factorIndex]
    }
  })
  const factorVariance = factorContribution.reduce((sum, p) => sum + p.y, 0)
  const specificVariance = parsed.weights.reduce((sum, w, i) => sum + (w * parsed.specificVols[i]) ** 2, 0)
  const totalVariance = factorVariance + specificVariance
  const paths = Array.from({ length: 12 }, (_, i) => ({
    name: `Path ${i + 1}`,
    points: simulateFactorPath(parsed, factorChol, 30)
  }))
  const varLoss = quantile(losses, parsed.alpha)
  return {
    meanLoss: -dot(parsed.weights, parsed.exposures.map(row => dot(row, parsed.factorMeans))),
    stdevLoss: Math.sqrt(dot(parsed.weights, matrixVector(assetCov, parsed.weights))),
    varLoss,
    esLoss: expectedShortfall(losses, varLoss),
    systematicShare: factorVariance / totalVariance,
    specificShare: specificVariance / totalVariance,
    factorContribution,
    losses,
    paths
  }
}

function renderResults(result) {
  document.getElementById('factor-results').innerHTML = `
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Expected loss</div><div class="metric-value">${fmt(result.meanLoss)}</div></div>
      <div class="metric"><div class="metric-label">Loss volatility</div><div class="metric-value">${fmt(result.stdevLoss)}</div></div>
      <div class="metric"><div class="metric-label">VaR</div><div class="metric-value">${fmt(result.varLoss)}</div></div>
      <div class="metric"><div class="metric-label">Expected shortfall</div><div class="metric-value">${fmt(result.esLoss)}</div></div>
      <div class="metric"><div class="metric-label">Systematic risk share</div><div class="metric-value">${fmt(result.systematicShare)}</div></div>
      <div class="metric"><div class="metric-label">Specific risk share</div><div class="metric-value">${fmt(result.specificShare)}</div></div>
    </div>
    <p class="small">The histogram shows one-horizon factor-model losses. The path chart below shows how repeated common-factor and specific shocks can push the portfolio over time.</p>`

  renderHistogram(document.getElementById('factor-histogram'), result.losses, [
    { value: result.varLoss, label: 'VaR', color: '#1d6b2f' },
    { value: result.esLoss, label: 'ES', color: '#a33a00' }
  ], 'Factor-model loss distribution')

  renderMultiLineChart(document.getElementById('factor-paths'), result.paths, 'Illustrative factor-driven portfolio paths')
  renderLineChart(document.getElementById('factor-contrib'), result.factorContribution, 'Factor variance contribution', 'F')
}

function updatePython(inputs) {
  document.getElementById('factor-python').textContent = buildPython(inputs)
}

function hasAnyValue(inputs) {
  return Object.values(inputs).some(input => input.value.trim() !== '')
}

function updateCanRun(inputs, showErrors = false) {
  const validation = validate(inputs)
  const anyValue = hasAnyValue(inputs)
  const canRun = Object.values(inputs).every(input => input.value.trim() !== '') && validation.errors.length === 0
  document.getElementById('factor-run').disabled = !canRun
  document.getElementById('factor-errors').innerHTML = showErrors && anyValue ? validation.errors.map(error => `<div class="notice">${error}</div>`).join('') : ''
  updatePython(inputs)
  return validation
}

function clearOutputs() {
  document.getElementById('factor-results').innerHTML = '<div class="empty">Enter inputs and run the model to generate results.</div>'
  document.getElementById('factor-histogram').innerHTML = ''
  document.getElementById('factor-paths').innerHTML = ''
  document.getElementById('factor-contrib').innerHTML = ''
}

export function setupFactorPage() {
  const ids = ['assetCount', 'factorCount', 'weights', 'factorMeans', 'factorVols', 'specificVols', 'exposures', 'corr', 'alpha', 'simulations']
  const inputs = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]))
  ids.forEach(id => inputs[id].addEventListener('input', () => updateCanRun(inputs, hasAnyValue(inputs))))
  document.getElementById('factor-run').addEventListener('click', () => {
    const validation = updateCanRun(inputs, true)
    if (validation.errors.length) return
    const result = runModel(validation.parsed)
    renderResults(result)
    document.getElementById('factor-errors').innerHTML = '<div class="success">Model ran successfully.</div>'
  })
  document.getElementById('factor-clear').addEventListener('click', () => {
    ids.forEach(id => { inputs[id].value = '' })
    clearOutputs()
    updateCanRun(inputs, false)
  })
  clearOutputs()
  updateCanRun(inputs, false)
}
