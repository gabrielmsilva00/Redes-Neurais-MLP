const themeToggle = document.querySelector('.theme-toggle');
const themeOptions = document.querySelectorAll('.theme-option');
const htmlElement = document.documentElement;

function setTheme(themeName) {
  htmlElement.classList.remove('light-theme', 'dark-theme', 'black-theme');
  htmlElement.classList.add(`${themeName}-theme`);
  localStorage.setItem('theme', themeName);
  updatePlotlyTheme(themeName);
}

function loadTheme() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    setTheme(savedTheme);
  } else {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  }
}

themeOptions.forEach(option => {
  option.addEventListener('click', () => {
    setTheme(option.dataset.theme);
  });
});

function updatePlotlyTheme(themeName) {
  let bgColor, gridColor, textColor;
  
  switch (themeName) {
    case 'light':
      bgColor = '#ffffff';
      gridColor = '#e1e1e1';
      textColor = '#000000';
      break;
    case 'dark':
      bgColor = '#1e1e1e';
      gridColor = '#2a2a2a';
      textColor = '#ffffff';
      break;
    case 'black':
      bgColor = '#000000';
      gridColor = '#1a1a1a';
      textColor = '#ffffff';
      break;
  }
  
  const plotlyConfig = {
    layout: {
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
      font: { color: textColor },
      xaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
      yaxis: { gridcolor: gridColor, zerolinecolor: gridColor }
    }
  };
  
  const charts = ['loss-chart', 'accuracy-chart', 'all-folds-chart'];
  
  charts.forEach(chartId => {
    const chartElem = document.getElementById(chartId);
    if (chartElem && chartElem._fullData && chartElem._fullLayout) {
      Plotly.relayout(chartElem, plotlyConfig.layout);
    }
  });
}

const trainBtn = document.getElementById('train-btn');
const nSamplesInput = document.getElementById('n-samples');
const nFeaturesInput = document.getElementById('n-features');
const epochsInput = document.getElementById('epochs');
const batchSizeInput = document.getElementById('batch-size');
const kFoldsInput = document.getElementById('k-folds');
const learningRateInput = document.getElementById('learning-rate');
const dropoutRateInput = document.getElementById('dropout-rate');
const hiddenLayersInput = document.getElementById('hidden-layers');
const activationFuncInput = document.getElementById('activation-func');
const optimizerInput = document.getElementById('optimizer');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status-text');
const logElement = document.getElementById('log');
const lossChart = document.getElementById('loss-chart');
const accuracyChart = document.getElementById('accuracy-chart');
const allFoldsChart = document.getElementById('all-folds-chart');
const metricsElement = document.getElementById('metrics');
const viewModeBtns = document.querySelectorAll('.view-mode-btn');

let models = [];
let currentBestModel = null;
let foldResults = [];
let isTraining = false;
const logMessages = [];

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  const formattedMessage = `[${timestamp}] ${message}`;
  console.log(formattedMessage);
  
  logMessages.push(formattedMessage);
  if (logMessages.length > 100) logMessages.shift();
  
  logElement.innerHTML = logMessages.join('<br>');
  logElement.scrollTop = logElement.scrollHeight;
}

function updateProgress(value, message = null) {
  const percent = Math.round(value * 100);
  progressBar.style.width = `${percent}%`;
  progressText.textContent = `${percent}%`;
  
  if (message) {
    statusText.textContent = message;
  }
}

function generateData(nSamples, nFeatures) {
  const xData = [];
  const yData = [];
  
  for (let i = 0; i < nSamples; i++) {
    const features = [];
    for (let j = 0; j < nFeatures; j++) {
      features.push(Math.random() * 2 - 1);
    }
    xData.push(features);
    
    const sum = features.slice(0, 3).reduce((a, b) => a + b, 0);
    const target = sum > 0 ? 1 : 0;
    yData.push(target);
  }
  
  return {
    x: tf.tensor2d(xData),
    y: tf.tensor2d(yData, [nSamples, 1])
  };
}

function getKFolds(x, y, k) {
  const numSamples = x.shape[0];
  const foldSize = Math.floor(numSamples / k);
  const folds = [];
  
  const indices = Array.from(Array(numSamples).keys());
  tf.util.shuffle(indices);
  
  for (let i = 0; i < k; i++) {
    const start = i * foldSize;
    const end = (i + 1) * foldSize;
    
    const validIndices = indices.slice(start, end);
    const trainIndices = indices.filter(idx => !validIndices.includes(idx));
    
    const xTrain = tf.gather(x, trainIndices);
    const yTrain = tf.gather(y, trainIndices);
    const xValid = tf.gather(x, validIndices);
    const yValid = tf.gather(y, validIndices);
    
    folds.push({ xTrain, yTrain, xValid, yValid });
  }
  
  return folds;
}

function createModel(inputShape, params) {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    inputShape: [inputShape],
    units: params.hiddenLayers[0],
    activation: params.activation,
    kernelInitializer: 'heNormal'
  }));
  
  if (params.dropoutRate > 0) {
    model.add(tf.layers.dropout({ rate: params.dropoutRate }));
  }
  
  for (let i = 1; i < params.hiddenLayers.length; i++) {
    model.add(tf.layers.dense({
      units: params.hiddenLayers[i],
      activation: params.activation
    }));
    
    if (params.dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: params.dropoutRate }));
    }
  }
  
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  
  let optimizer;
  switch (params.optimizer) {
    case 'sgd':
      optimizer = tf.train.sgd(params.learningRate);
      break;
    case 'rmsprop':
      optimizer = tf.train.rmsprop(params.learningRate);
      break;
    case 'adagrad':
      optimizer = tf.train.adagrad(params.learningRate);
      break;
    default:
      optimizer = tf.train.adam(params.learningRate);
  }
  
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

async function trainModel() {
  if (isTraining) return;
  
  isTraining = true;
  trainBtn.disabled = true;
  
  try {
    const nSamples = parseInt(nSamplesInput.value);
    const nFeatures = parseInt(nFeaturesInput.value);
    const epochs = parseInt(epochsInput.value);
    const batchSize = parseInt(batchSizeInput.value);
    const kFolds = parseInt(kFoldsInput.value) || 5;
    const learningRate = parseFloat(learningRateInput.value);
    const dropoutRate = parseFloat(dropoutRateInput.value);
    const hiddenLayersStr = hiddenLayersInput.value;
    const hiddenLayers = hiddenLayersStr.split(',').map(n => parseInt(n.trim()));
    const activation = activationFuncInput.value;
    const optimizer = optimizerInput.value;
    
    log(`Iniciando treinamento com ${kFolds} folds`);
    
    const params = {
      learningRate,
      dropoutRate,
      hiddenLayers,
      activation,
      optimizer,
      epochs,
      batchSize
    };
    
    log(`Gerando ${nSamples} amostras com ${nFeatures} features...`);
    const data = generateData(nSamples, nFeatures);
    
    log(`Dividindo dados em ${kFolds} folds`);
    const folds = getKFolds(data.x, data.y, kFolds);
    
    models = [];
    foldResults = [];
    
    let bestValAcc = -1;
    let bestFoldIndex = -1;
    
    for (let fold = 0; fold < kFolds; fold++) {
      log(`Treinando modelo no fold ${fold + 1}/${kFolds}`);
      updateProgress((fold / kFolds), `Treinando fold ${fold + 1}/${kFolds}`);
      
      const foldData = folds[fold];
      const model = createModel(nFeatures, params);
      
      const trainLogs = [];
      
      await model.fit(foldData.xTrain, foldData.yTrain, {
        epochs: epochs,
        batchSize: batchSize,
        validationData: [foldData.xValid, foldData.yValid],
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            trainLogs.push({
              epoch: epoch,
              loss: logs.loss,
              acc: logs.acc,
              val_loss: logs.val_loss,
              val_acc: logs.val_acc
            });
            
            const progress = (fold / kFolds) + ((epoch + 1) / epochs / kFolds);
            updateProgress(progress, `Fold ${fold + 1}/${kFolds}, Época ${epoch + 1}/${epochs}`);
          }
        }
      });
      
      const evalResult = model.evaluate(foldData.xValid, foldData.yValid);
      const valLoss = evalResult[0].dataSync()[0];
      const valAcc = evalResult[1].dataSync()[0];
      
      log(`Fold ${fold + 1} - Perda: ${valLoss.toFixed(4)}, Acurácia: ${(valAcc * 100).toFixed(2)}%`);
      
      models.push(model);
      foldResults.push({
        fold: fold,
        trainLogs: trainLogs,
        valLoss: valLoss,
        valAcc: valAcc
      });
      
      if (valAcc > bestValAcc) {
        bestValAcc = valAcc;
        bestFoldIndex = fold;
        currentBestModel = model;
      }
    }
    
    log(`Treinamento completo. Melhor fold: ${bestFoldIndex + 1} com acurácia: ${(bestValAcc * 100).toFixed(2)}%`);
    updateProgress(1, `Treinamento concluído! Melhor acurácia: ${(bestValAcc * 100).toFixed(2)}%`);
    
    displayResults(bestFoldIndex);
    
  } catch (error) {
    log(`ERRO: ${error.message}`);
    statusText.textContent = 'Erro durante o treinamento. Veja o log para detalhes.';
  } finally {
    isTraining = false;
    trainBtn.disabled = false;
  }
}

function displayResults(bestFoldIndex) {
  const bestFold = foldResults[bestFoldIndex];
  
  const epochs = bestFold.trainLogs.map(log => log.epoch);
  const trainLoss = bestFold.trainLogs.map(log => log.loss);
  const trainAcc = bestFold.trainLogs.map(log => log.acc);
  const valLoss = bestFold.trainLogs.map(log => log.val_loss);
  const valAcc = bestFold.trainLogs.map(log => log.val_acc);
  
  const lossData = [
    { x: epochs, y: trainLoss, type: 'scatter', mode: 'lines', name: 'Perda (Treino)' },
    { x: epochs, y: valLoss, type: 'scatter', mode: 'lines', name: 'Perda (Validação)' }
  ];
  
  const accData = [
    { x: epochs, y: trainAcc, type: 'scatter', mode: 'lines', name: 'Acurácia (Treino)' },
    { x: epochs, y: valAcc, type: 'scatter', mode: 'lines', name: 'Acurácia (Validação)' }
  ];
  
  const layout = {
    title: `Melhor Modelo (Fold ${bestFoldIndex + 1})`,
    xaxis: { title: 'Época' },
    yaxis: { title: '' }
  };
  
  Plotly.newPlot('loss-chart', lossData, {...layout, yaxis: { title: 'Perda' }});
  Plotly.newPlot('accuracy-chart', accData, {...layout, yaxis: { title: 'Acurácia' }});
  
  displayAllFoldsResults();
  displayMetricsTable();
  
  document.getElementById('best-model-view').style.display = 'block';
  document.getElementById('all-folds-view').style.display = 'none';
}

function displayAllFoldsResults() {
  const allFoldsData = [];
  
  foldResults.forEach((result, idx) => {
    allFoldsData.push({
      x: result.trainLogs.map(log => log.epoch),
      y: result.trainLogs.map(log => log.val_acc),
      type: 'scatter',
      mode: 'lines',
      name: `Fold ${idx + 1}`
    });
  });
  
  const layout = {
    title: 'Acurácia de Validação em Todos os Folds',
    xaxis: { title: 'Época' },
    yaxis: { title: 'Acurácia de Validação' }
  };
  
  Plotly.newPlot('all-folds-chart', allFoldsData, layout);
}

function displayMetricsTable() {
  const allMetrics = foldResults.map((result, idx) => {
    return {
      fold: idx + 1,
      valLoss: result.valLoss.toFixed(4),
      valAcc: (result.valAcc * 100).toFixed(2) + '%'
    };
  });
  
  const avgLoss = allMetrics.reduce((sum, curr) => sum + parseFloat(curr.valLoss), 0) / allMetrics.length;
  const avgAcc = allMetrics.reduce((sum, curr) => sum + parseFloat(curr.valAcc), 0) / allMetrics.length;
  
  let metricsHTML = `
    <h3>Métricas de Validação Cruzada</h3>
    <table>
      <thead>
        <tr>
          <th>Fold</th>
          <th>Perda</th>
          <th>Acurácia</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  allMetrics.forEach(metric => {
    metricsHTML += `
      <tr>
        <td>${metric.fold}</td>
        <td>${metric.valLoss}</td>
        <td>${metric.valAcc}</td>
      </tr>
    `;
  });
  
  metricsHTML += `
      <tr class="highlight">
        <td><strong>Média</strong></td>
        <td><strong>${avgLoss.toFixed(4)}</strong></td>
        <td><strong>${avgAcc.toFixed(2)}%</strong></td>
      </tr>
    </tbody>
    </table>
  `;
  
  metricsElement.innerHTML = metricsHTML;
}

function switchView(mode) {
  if (mode === 'best') {
    document.getElementById('best-model-view').style.display = 'block';
    document.getElementById('all-folds-view').style.display = 'none';
  } else {
    document.getElementById('best-model-view').style.display = 'none';
    document.getElementById('all-folds-view').style.display = 'block';
  }
  
  viewModeBtns.forEach(btn => {
    if (btn.dataset.mode === mode) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function init() {
  if (typeof tf === 'undefined') {
    log('ERRO: TensorFlow.js não carregado!');
    statusText.textContent = 'ERRO: TensorFlow.js não disponível';
    trainBtn.disabled = true;
    return;
  }
  
  loadTheme();
  statusText.textContent = 'Pronto para iniciar treinamento.';
  trainBtn.addEventListener('click', trainModel);
  
  viewModeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      switchView(btn.dataset.mode);
    });
  });
}

window.addEventListener('load', init);