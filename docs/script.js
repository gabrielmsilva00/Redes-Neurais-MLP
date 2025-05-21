const themeToggle = document.querySelector('.theme-toggle');
const themeOptions = document.querySelectorAll('.theme-option');
const htmlElement = document.documentElement;

function setTheme(themeName) {
  htmlElement.classList.remove('light-theme', 'dark-theme', 'black-theme');
  htmlElement.classList.add(`${themeName}-theme`);
  localStorage.setItem('theme', themeName);
  updatePlotlyChartsTheme(); 
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

function getPlotlyColorsFromCSS() {
  const computedStyle = getComputedStyle(document.documentElement);
  return {
    paper_bgcolor: computedStyle.getPropertyValue('--plotly-paper-bg-color').trim() || '#ffffff',
    plot_bgcolor: computedStyle.getPropertyValue('--plotly-plot-bg-color').trim() || '#ffffff',
    font_color: computedStyle.getPropertyValue('--plotly-font-color').trim() || '#000000',
    title_color: computedStyle.getPropertyValue('--plotly-title-font-color').trim() || computedStyle.getPropertyValue('--plotly-font-color').trim() || '#000000', 
    gridcolor: computedStyle.getPropertyValue('--plotly-grid-color').trim() || '#e1e1e1',
    linecolor: computedStyle.getPropertyValue('--plotly-line-color').trim() || '#d3d3d3', 
    zerolinecolor: computedStyle.getPropertyValue('--plotly-zeroline-color').trim() || computedStyle.getPropertyValue('--plotly-grid-color').trim() || '#e1e1e1', 
    legend_bgcolor: computedStyle.getPropertyValue('--plotly-legend-bg-color').trim() || 'rgba(255,255,255,0.5)',
    legend_font_color: computedStyle.getPropertyValue('--plotly-legend-font-color').trim() || computedStyle.getPropertyValue('--plotly-font-color').trim() || '#000000',
    legend_bordercolor: computedStyle.getPropertyValue('--plotly-legend-border-color').trim() || '#cccccc'
  };
}

function getCurrentPlotlyThemeLayout() {
    const colors = getPlotlyColorsFromCSS();
    return {
        paper_bgcolor: colors.paper_bgcolor,
        plot_bgcolor: colors.plot_bgcolor,
        font: { color: colors.font_color },
        titlefont: { color: colors.title_color }, 
        xaxis: {
            gridcolor: colors.gridcolor,
            zerolinecolor: colors.zerolinecolor,
            linecolor: colors.linecolor,
            tickfont: { color: colors.font_color },
            titlefont: { color: colors.font_color } 
        },
        yaxis: {
            gridcolor: colors.gridcolor,
            zerolinecolor: colors.zerolinecolor,
            linecolor: colors.linecolor,
            tickfont: { color: colors.font_color },
            titlefont: { color: colors.font_color } 
        },
        legend: {
          bgcolor: colors.legend_bgcolor,
          font: { color: colors.legend_font_color },
          bordercolor: colors.legend_bordercolor,
          borderwidth: 1
        },

        template: {}
    };
}

function updatePlotlyChartsTheme() {
  const newLayout = getCurrentPlotlyThemeLayout();
  const charts = ['loss-chart', 'accuracy-chart', 'all-folds-chart'];

  charts.forEach(chartId => {
    const chartElem = document.getElementById(chartId);

    if (chartElem && chartElem.data && chartElem.classList.contains('js-plotly-plot')) {
      Plotly.relayout(chartElem, {
        'paper_bgcolor': newLayout.paper_bgcolor,
        'plot_bgcolor': newLayout.plot_bgcolor,
        'font.color': newLayout.font.color,
        'titlefont.color': newLayout.titlefont.color,
        'xaxis.gridcolor': newLayout.xaxis.gridcolor,
        'xaxis.zerolinecolor': newLayout.xaxis.zerolinecolor,
        'xaxis.linecolor': newLayout.xaxis.linecolor,
        'xaxis.tickfont.color': newLayout.xaxis.tickfont.color,
        'xaxis.titlefont.color': newLayout.xaxis.titlefont.color,
        'yaxis.gridcolor': newLayout.yaxis.gridcolor,
        'yaxis.zerolinecolor': newLayout.yaxis.zerolinecolor,
        'yaxis.linecolor': newLayout.yaxis.linecolor,
        'yaxis.tickfont.color': newLayout.yaxis.tickfont.color,
        'yaxis.titlefont.color': newLayout.yaxis.titlefont.color,
        'legend.bgcolor': newLayout.legend.bgcolor,
        'legend.font.color': newLayout.legend.font.color,
        'legend.bordercolor': newLayout.legend.bordercolor,
        'template': newLayout.template 
      });
    }
  });
}

const trainBtn = document.getElementById('train-btn');
const saveModelBtn = document.getElementById('save-model-btn');
const loadModelBtn = document.getElementById('load-model-btn');
const loadModelJsonInput = document.getElementById('load-model-json');
const loadModelWeightsInput = document.getElementById('load-model-weights');

const nSamplesInput = document.getElementById('n-samples');
const nFeaturesInput = document.getElementById('n-features');
const nSplitsInput = document.getElementById('n-splits');
const testSizeInput = document.getElementById('test-size');
const classWeights0Input = document.getElementById('class-weights-0');
const classWeights1Input = document.getElementById('class-weights-1');

const maxEpochsInput = document.getElementById('max-epochs');
const batchSizeInput = document.getElementById('batch-size');
const patienceInput = document.getElementById('patience');
const learningRateInput = document.getElementById('learning-rate');
const optimizerInput = document.getElementById('optimizer');

const dynamicLayersContainer = document.getElementById('dynamic-layers-container');
const addHiddenLayerBtn = document.getElementById('add-hidden-layer-btn');
const removeHiddenLayerBtn = document.getElementById('remove-hidden-layer-btn');
const outputUnitsInput = document.getElementById('output-units');
const outputActivationInput = document.getElementById('output-activation');

const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status-text');
const logElement = document.getElementById('log'); 
const lossChartEl = document.getElementById('loss-chart');
const accuracyChartEl = document.getElementById('accuracy-chart');
const allFoldsChartEl = document.getElementById('all-folds-chart');
const metricsElement = document.getElementById('metrics');
const viewModeBtns = document.querySelectorAll('.view-mode-btn');
const stopTrainBtn = document.getElementById('stop-train-btn');

let currentBestModel = null;
let foldResults = [];
let isTraining = false;
const logMessages = [];
let globalData = { x: null, y: null };
let hiddenLayerCounter = 0;
let stopTrainingFlag = false;

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  const formattedMessage = `[${timestamp}] ${message}`;
  console.log(formattedMessage);

  const logLine = document.createElement('div'); 
  logLine.textContent = formattedMessage;
  logElement.appendChild(logLine);

  while (logElement.children.length > 200) { 
    logElement.removeChild(logElement.firstChild);
  }
  logElement.scrollTop = logElement.scrollHeight;
}

function updateProgress(value, message = null) {
  const percent = Math.round(value * 100);
  progressBar.style.width = `${percent}%`;
  progressText.textContent = `${percent}%`;
  if (message) statusText.textContent = message;
}

function generateData(nSamples, nFeatures) {
    return tf.tidy(() => {
        const xData = [];
        const yData = [];
        for (let i = 0; i < nSamples; i++) {
            const features = [];
            for (let j = 0; j < nFeatures; j++) features.push(Math.random() * 2 - 1);
            xData.push(features);
            const sum = features.slice(0, Math.min(3, nFeatures)).reduce((a, b) => a + b, 0);
            yData.push(sum > 0 ? 1 : 0);
        }
        return {
            x: tf.tensor2d(xData, [nSamples, nFeatures]),
            y: tf.tensor2d(yData, [nSamples, 1])
        };
    });
}

function getKFolds(x, y, k, testSize) {
    return tf.tidy(() => {
        const numSamples = x.shape[0];
        const indicesTensor = tf.tensor1d(Array.from(tf.util.createShuffledIndices(numSamples)), 'int32');

        const xShuffled = x.gather(indicesTensor);
        const yShuffled = y.gather(indicesTensor);

        const folds = [];

        if (k === 1) {
            const numTestSamples = Math.floor(numSamples * testSize);
            const numTrainSamples = numSamples - numTestSamples;
            folds.push({
                train: { x: xShuffled.slice(0, numTrainSamples), y: yShuffled.slice(0, numTrainSamples) },
                val: { x: xShuffled.slice(numTrainSamples), y: yShuffled.slice(numTrainSamples) }
            });
        } else {
            const foldSize = Math.floor(numSamples / k);
            for (let i = 0; i < k; i++) {
                const valStart = i * foldSize;
                const valEnd = (i + 1) * foldSize;
                const xVal = xShuffled.slice(valStart, valEnd - valStart);
                const yVal = yShuffled.slice(valStart, valEnd - valStart);
                const xTrainParts = [], yTrainParts = [];
                if (i > 0) { xTrainParts.push(xShuffled.slice(0, valStart)); yTrainParts.push(yShuffled.slice(0, valStart)); }
                if (i < k - 1) { xTrainParts.push(xShuffled.slice(valEnd)); yTrainParts.push(yShuffled.slice(valEnd)); }

                let xTrain, yTrain;
                if (xTrainParts.length === 0) {
                    xTrain = xShuffled.gather(tf.tensor1d([], 'int32'));
                    yTrain = yShuffled.gather(tf.tensor1d([], 'int32'));
                } else if (xTrainParts.length === 1) {
                    xTrain = xTrainParts[0];
                    yTrain = yTrainParts[0];
                } else {
                    xTrain = tf.concat(xTrainParts, 0);
                    yTrain = tf.concat(yTrainParts, 0);
                }
                folds.push({ train: { x: xTrain, y: yTrain }, val: { x: xVal, y: yVal } });
            }
        }
        indicesTensor.dispose();

        return folds;
    });
}

function getLayerDefinitionsFromUI() {
    const layerDefs = [];
    const hiddenLayerCards = dynamicLayersContainer.querySelectorAll('.layer-card.dynamic-hidden-layer');
    hiddenLayerCards.forEach((card, index) => {
        const units = parseInt(card.querySelector('.layer-units-input').value);
        const activation = card.querySelector('.layer-activation-input').value;
        if (!isNaN(units) && units > 0) {
            layerDefs.push({ units, activation, name: `hidden_layer_${index + 1}` });
        }
    });

    const outputUnits = parseInt(outputUnitsInput.value);
    const outputActivation = outputActivationInput.value;
    if(!isNaN(outputUnits) && outputUnits > 0) {
        layerDefs.push({ units: outputUnits, activation: outputActivation, name: 'output_layer' });
    }
    return layerDefs;
}

function createModel(inputShape, layerDefs, optimizerName, learningRateVal) {
  const model = tf.sequential();
  if (layerDefs.length === 0) {
      log("ERRO: Nenhuma camada definida para o modelo.");
      return null;
  }
  layerDefs.forEach((def, index) => {
    const layerConfig = { units: def.units, activation: def.activation, name: def.name };
    if (index === 0) {
      layerConfig.inputShape = [inputShape];
    }
    model.add(tf.layers.dense(layerConfig));
  });

  let optimizerInstance;
  if (optimizerName === 'adam') optimizerInstance = tf.train.adam(learningRateVal);
  else if (optimizerName === 'sgd') optimizerInstance = tf.train.sgd(learningRateVal);
  else if (optimizerName === 'rmsprop') optimizerInstance = tf.train.rmsprop(learningRateVal);
  else if (optimizerName === 'adagrad') optimizerInstance = tf.train.adagrad(learningRateVal);
  else optimizerInstance = tf.train.adam(learningRateVal);

  model.compile({
    optimizer: optimizerInstance,
    loss: layerDefs[layerDefs.length-1].activation === 'softmax' ? 'categoricalCrossentropy' : 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

class CustomLogger extends tf.Callback {
  constructor(foldIdentifier, maxEpochsPerFold, currentFoldNum, totalFolds, getStopSignal) {
    super();
    this.foldIdentifier = foldIdentifier;
    this.maxEpochsPerFold = maxEpochsPerFold;
    this.currentFoldNum = currentFoldNum;
    this.totalFolds = totalFolds;
    this.getStopSignal = getStopSignal;
  }
  async onEpochEnd(epoch, logs) {
    let trainingShouldStop = false;
    if (this.getStopSignal && this.getStopSignal()) {
      log(`Sinal de interrupção recebido na Época ${epoch + 1} do ${this.foldIdentifier}. Model.stopTraining será definido como true.`);
      if (this.model) {
        this.model.stopTraining = true;
      }
      trainingShouldStop = true;
    }
    const overallProgress = (this.currentFoldNum + (epoch + 1) / this.maxEpochsPerFold) / this.totalFolds;
    const valLoss = logs.val_loss !== undefined ? logs.val_loss.toFixed(4) : 'N/A';
    const valAccDisplay = logs.val_acc !== undefined ? (logs.val_acc * 100).toFixed(2) + '%' : 'N/A';

    updateProgress(overallProgress, `${this.foldIdentifier}, Época ${epoch + 1}: Perda Val: ${valLoss}, Acc Val: ${valAccDisplay}`);

    if (epoch % 10 === 0 || epoch === this.maxEpochsPerFold - 1 || trainingShouldStop) {
      const trainAccDisplay = logs.acc !== undefined ? (logs.acc * 100).toFixed(2) + '%' : 'N/A';
      log(`Época ${epoch + 1} (${this.foldIdentifier}): Perda Treino: ${logs.loss.toFixed(4)}, Acc Treino: ${trainAccDisplay}, Perda Val: ${valLoss}, Acc Val: ${valAccDisplay}${trainingShouldStop ? ' (Interrompendo...)' : ''}`);
    }

    await tf.nextFrame();
  }
}

async function trainModel() {
  if (isTraining) { log("Treinamento já está em andamento."); return; }
  isTraining = true;
  stopTrainingFlag = false;

  trainBtn.innerHTML = '<span class="material-icons">hourglass_top</span> Treinando...';
  trainBtn.disabled = true;
  stopTrainBtn.style.display = 'inline-flex'; 
  stopTrainBtn.disabled = false;
  saveModelBtn.disabled = true;
  loadModelBtn.disabled = true;
  log("Iniciando processo de treinamento...");
  updateProgress(0, "Preparando dados e modelo...");
  await tf.nextFrame();
  try {
    const nSamples = parseInt(nSamplesInput.value);
    const nFeatures = parseInt(nFeaturesInput.value);
    const nSplits = parseInt(nSplitsInput.value);
    const testSize = parseFloat(testSizeInput.value);
    const classWeight0 = parseFloat(classWeights0Input.value);
    const classWeight1 = parseFloat(classWeights1Input.value);
    const maxEpochs = parseInt(maxEpochsInput.value);
    const batchSize = parseInt(batchSizeInput.value);
    const patience = parseInt(patienceInput.value);
    const learningRateVal = parseFloat(learningRateInput.value);
    const optimizerName = optimizerInput.value;
    const layerDefs = getLayerDefinitionsFromUI();
    if (layerDefs.length === 0) {
        log("ERRO: Arquitetura da rede não definida ou inválida.");
        throw new Error("Arquitetura da rede não definida.");
    }
    log("Definições das camadas: " + JSON.stringify(layerDefs));
    await tf.setBackend('cpu'); 
    log(`Backend TensorFlow.js: ${tf.getBackend()}`);

    if (globalData.x) { tf.dispose(globalData.x); globalData.x = null; }
    if (globalData.y) { tf.dispose(globalData.y); globalData.y = null; }
    globalData = generateData(nSamples, nFeatures);
    log(`Dados gerados: ${nSamples} amostras, ${nFeatures} features.`);
    await tf.nextFrame();

    const foldsData = getKFolds(globalData.x, globalData.y, nSplits, testSize);
    log(`Dados divididos em ${foldsData.length} folds.`);
    await tf.nextFrame();

    if (currentBestModel) { currentBestModel.dispose(); currentBestModel = null;}
    foldResults = [];
    let bestFoldIndex = -1;
    let bestValAccuracyOverall = -1; 

    for (let i = 0; i < foldsData.length; i++) {
      if (stopTrainingFlag) {
          log(`Treinamento interrompido pelo usuário antes do Fold ${i + 1}.`);
          break;
      }
      updateProgress(i / foldsData.length, `Preparando Fold ${i + 1}/${foldsData.length}...`);
      await tf.nextFrame();

      log(`--- Fold ${i + 1} ---`);

      const { train, val } = foldsData[i];
      if (!train.x || !train.y || !val.x || !val.y || train.x.shape[0] === 0 || val.x.shape[0] === 0) {
          log(`Fold ${i+1} ignorado devido a dados de treino/validação insuficientes. Train X: ${train.x ? train.x.shape : 'null'}, Val X: ${val.x ? val.x.shape : 'null'}`);

          tf.dispose([train.x, train.y, val.x, val.y].filter(t => t));
          continue;
      }
      const model = createModel(nFeatures, layerDefs, optimizerName, learningRateVal);
      if (!model) {
          log(`Falha ao criar modelo para o Fold ${i+1}.`);
          tf.dispose([train.x, train.y, val.x, val.y].filter(t => t));
          continue;
      }

      const customLogger = new CustomLogger(`Fold ${i + 1}`, maxEpochs, i, foldsData.length, () => stopTrainingFlag);
      let history;
      try {
        history = await model.fit(train.x, train.y, {
          epochs: maxEpochs,
          batchSize: batchSize,
          validationData: [val.x, val.y],
          classWeight: {0: classWeight0, 1: classWeight1},
          callbacks: [
            tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: patience, verbose: 1}),
            customLogger
          ]
        });
      } catch (fitError) {
          log(`Erro durante model.fit() para o Fold ${i+1}: ${fitError.message}`);
          if (model) model.dispose();
          tf.dispose([train.x, train.y, val.x, val.y].filter(t => t));
          continue;
      }

      if (stopTrainingFlag && model.stopTraining) {
          log(`Fold ${i + 1} interrompido durante o treinamento.`);
      }

      const finalValMetrics = model.evaluate(val.x, val.y, {batchSize: batchSize});
      const valLoss = (Array.isArray(finalValMetrics) ? finalValMetrics[0] : finalValMetrics).dataSync()[0];
      const valAcc = (Array.isArray(finalValMetrics) ? finalValMetrics[1] : finalValMetrics).dataSync()[0];
      tf.dispose(finalValMetrics);
      log(`Fold ${i + 1} - Validação Final: Perda = ${valLoss.toFixed(4)}, Acurácia = ${(valAcc * 100).toFixed(2)}%`);

      if (history && history.history) {
          foldResults.push({
              loss: history.history.loss || [],
              accuracy: history.history.acc || [],
              val_loss: history.history.val_loss || [],
              val_accuracy: history.history.val_acc || [],
              final_val_loss: valLoss,
              final_val_accuracy: valAcc
          });
      } else if (!stopTrainingFlag) {
          log(`AVISO: Fold ${i+1} completou model.fit() mas não produziu histórico válido.`);
           foldResults.push({ final_val_loss: valLoss, final_val_accuracy: valAcc }); 
      }

      if (valAcc > bestValAccuracyOverall && !stopTrainingFlag) {
        bestValAccuracyOverall = valAcc;
        bestFoldIndex = i;
        if (currentBestModel) currentBestModel.dispose();
        currentBestModel = model; 
      } else {
        model.dispose(); 
      }
      tf.dispose([train.x, train.y, val.x, val.y].filter(t => t)); 
    }

    if (globalData.x) { tf.dispose(globalData.x); globalData.x = null; }
    if (globalData.y) { tf.dispose(globalData.y); globalData.y = null; }

    if (stopTrainingFlag) {
        log("Processo de treinamento interrompido pelo usuário.");
        updateProgress( (foldResults.length / (foldsData.length || 1) ), "Treinamento interrompido.");
    } else if (foldResults.length > 0) {
        log("Treinamento concluído para todos os folds processados.");
        updateProgress(1, "Treinamento concluído.");
    } else {
        log("Nenhum fold foi treinado com sucesso ou o treinamento foi interrompido muito cedo.");
        updateProgress(0, "Treinamento finalizado sem resultados de folds.");
    }

    if (currentBestModel && bestFoldIndex !== -1 && foldResults[bestFoldIndex]) {
        log(`Melhor modelo (Fold ${bestFoldIndex + 1}) com acurácia de validação de ${(bestValAccuracyOverall * 100).toFixed(2)}% mantido.`);
        plotBestModelResults(foldResults[bestFoldIndex]);
    } else if (foldResults.length > 0 && foldResults[foldResults.length -1]){ 
        log("Exibindo resultados do último fold processado (ou único).");
        plotBestModelResults(foldResults[foldResults.length -1]);
    } else {
        log("Nenhum resultado de fold para exibir.");
         plotBestModelResults(null);
    }
    plotAllFoldsResults(foldResults);
    displayMetrics(foldResults);
  } catch (error) {
    log(`ERRO GERAL no Treinamento: ${error.message}\nStack: ${error.stack}`);
    updateProgress(0, `Erro: ${error.message.substring(0, 50)}...`);
  } finally {
    isTraining = false;
    trainBtn.innerHTML = '<span class="material-icons">play_arrow</span> Executar Treinamento';
    trainBtn.disabled = false;
    stopTrainBtn.style.display = 'none';
    stopTrainBtn.disabled = true;
    saveModelBtn.disabled = !currentBestModel;
    loadModelBtn.disabled = false;

    log("Rotina de finalização do treinamento executada.");
    await tf.nextFrame();
  }
}

function plotBestModelResults(bestFoldData) {
  const baseLayout = getCurrentPlotlyThemeLayout(); 
  const responsiveConfig = {responsive: true};

  if (!bestFoldData || !bestFoldData.loss || bestFoldData.loss.length === 0 || !bestFoldData.val_loss || bestFoldData.val_loss.length === 0) {
    Plotly.purge(lossChartEl); Plotly.purge(accuracyChartEl);
    log("Dados insuficientes para plotar gráficos do melhor modelo.");

    Plotly.newPlot(lossChartEl, [], { ...baseLayout, title: 'Curvas de Perda (Sem Dados)'}, responsiveConfig);
    Plotly.newPlot(accuracyChartEl, [], { ...baseLayout, title: 'Curvas de Acurácia (Sem Dados)'}, responsiveConfig);
    return;
  }
  const epochs = Array.from({ length: bestFoldData.loss.length }, (_, i) => i + 1);

  const lossTraceTrain = { x: epochs, y: bestFoldData.loss, mode: 'lines', name: 'Perda (Treino)', line: {color: 'var(--plotly-trace-blue)'} }; 
  const lossTraceVal = { x: epochs, y: bestFoldData.val_loss, mode: 'lines', name: 'Perda (Validação)', line: {color: 'var(--plotly-trace-orange)'} };
  const accuracyTraceTrain = { x: epochs, y: bestFoldData.accuracy.map(a => a*100), mode: 'lines', name: 'Acurácia (Treino)', line: {color: 'var(--plotly-trace-green)'} };
  const accuracyTraceVal = { x: epochs, y: bestFoldData.val_accuracy.map(a => a*100), mode: 'lines', name: 'Acurácia (Validação)', line: {color: 'var(--plotly-trace-red)'} };

  Plotly.newPlot(lossChartEl, [lossTraceTrain, lossTraceVal], { ...baseLayout, title: 'Curvas de Perda (Melhor Fold)', yaxis: {...baseLayout.yaxis, title: 'Perda'} , xaxis: {...baseLayout.xaxis, title: 'Época'} }, responsiveConfig);
  Plotly.newPlot(accuracyChartEl, [accuracyTraceTrain, accuracyTraceVal], { ...baseLayout, title: 'Curvas de Acurácia (Melhor Fold)', yaxis: {...baseLayout.yaxis, title: 'Acurácia (%)'} , xaxis: {...baseLayout.xaxis, title: 'Época'} }, responsiveConfig);
  switchView('best');
}

function plotAllFoldsResults(results) {
  const baseLayout = getCurrentPlotlyThemeLayout();
  const responsiveConfig = {responsive: true};

  if (!results || results.length === 0) {
    Plotly.purge(allFoldsChartEl);
    Plotly.newPlot(allFoldsChartEl, [], { ...baseLayout, title: 'Perda de Validação (Todos Folds - Sem Dados)'}, responsiveConfig);
    return;
  }
  const tracesLoss = [];
  const colorPalette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

  results.forEach((foldData, index) => {
    if (foldData && foldData.val_loss && foldData.val_loss.length > 0) {
        const epochs = Array.from({ length: foldData.val_loss.length }, (_, i) => i + 1);
        tracesLoss.push({
            x: epochs,
            y: foldData.val_loss,
            mode: 'lines',
            name: `Fold ${index + 1} Val. Loss`,
            line: { color: colorPalette[index % colorPalette.length] }
        });
    }
  });

  if(tracesLoss.length > 0) {
    Plotly.newPlot(allFoldsChartEl, tracesLoss, { ...baseLayout, title: 'Perda de Validação (Todos Folds)', yaxis: {...baseLayout.yaxis, title: 'Perda'} , xaxis: {...baseLayout.xaxis, title: 'Época'} }, responsiveConfig);
  } else {
    Plotly.purge(allFoldsChartEl);
    Plotly.newPlot(allFoldsChartEl, [], { ...baseLayout, title: 'Perda de Validação (Todos Folds - Sem Dados)'}, responsiveConfig);
  }
}

function displayMetrics(results) {
  if (!results || results.length === 0) {
    metricsElement.innerHTML = '<p>Nenhuma métrica disponível. Execute o treinamento.</p>';
    return;
  }
  let metricsHTML = `<h3>Métricas Finais de Validação por Fold</h3>
                     <div class="table-responsive">
                       <table>
                         <thead><tr><th>Fold</th><th>Perda</th><th>Acurácia</th></tr></thead>
                         <tbody>`;
  let totalLoss = 0, totalAcc = 0, validFolds = 0;
  let bestFoldOverallAccuracy = -1, bestFoldOverallIndex = -1;

  results.forEach((fold, index) => {
    if (fold && typeof fold.final_val_loss === 'number' && typeof fold.final_val_accuracy === 'number') {
      const loss = fold.final_val_loss;
      const acc = fold.final_val_accuracy;
      totalLoss += loss;
      totalAcc += acc;
      validFolds++;
      if (acc > bestFoldOverallAccuracy) {
          bestFoldOverallAccuracy = acc;
          bestFoldOverallIndex = index;
      }
    }
  });

  results.forEach((fold, index) => {
    if (fold && typeof fold.final_val_loss === 'number' && typeof fold.final_val_accuracy === 'number') {
      const loss = fold.final_val_loss;
      const acc = fold.final_val_accuracy * 100;
      const isBest = index === bestFoldOverallIndex;
      metricsHTML += `<tr class="${isBest ? 'highlight-best-fold' : ''}"><td>${index + 1}</td><td>${loss.toFixed(4)}</td><td>${acc.toFixed(2)}%</td></tr>`;
    } else {
      metricsHTML += `<tr><td>${index + 1}</td><td colspan="2">Dados incompletos</td></tr>`;
    }
  });

  if (validFolds > 0) {
    const avgLoss = totalLoss / validFolds;
    const avgAcc = (totalAcc / validFolds) * 100;
    metricsHTML += `<tr class="summary-row"><td>Média</td><td>${avgLoss.toFixed(4)}</td><td>${avgAcc.toFixed(2)}%</td></tr>`;
  } else {
    metricsHTML += `<tr><td colspan="3" style="text-align:center;">Nenhum fold completou com sucesso.</td></tr>`;
  }
  metricsHTML += `</tbody></table></div>`;
  metricsElement.innerHTML = metricsHTML;
}

function switchView(mode) {
  document.getElementById('best-model-view').style.display = (mode === 'best' ? 'flex' : 'none'); 
  document.getElementById('all-folds-view').style.display = (mode === 'all' ? 'block' : 'none');
  viewModeBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));

  if (mode === 'best') {
    if (lossChartEl.data) Plotly.Plots.resize(lossChartEl);
    if (accuracyChartEl.data) Plotly.Plots.resize(accuracyChartEl);
  } else if (mode === 'all') {
    if (allFoldsChartEl.data) Plotly.Plots.resize(allFoldsChartEl);
  }
}

function addHiddenLayerUI(units = 16, activation = 'relu') {
    hiddenLayerCounter = dynamicLayersContainer.children.length; 
    const newLayerNum = hiddenLayerCounter + 1;

    const card = document.createElement('div');
    card.classList.add('layer-card', 'elevation-1', 'dynamic-hidden-layer');
    card.dataset.layerId = newLayerNum; 
    card.innerHTML = `
        <div class="layer-header">
            <h3>Camada Oculta ${newLayerNum}</h3>
        </div>
        <div class="form-group">
            <label for="layer-${newLayerNum}-units">Unidades:</label>
            <input type="number" id="layer-${newLayerNum}-units" value="${units}" min="1" max="1024" class="md-input layer-units-input">
        </div>
        <div class="form-group">
            <label for="layer-${newLayerNum}-activation">Ativação:</label>
            <select id="layer-${newLayerNum}-activation" class="md-input layer-activation-input">
                <option value="relu" ${activation === 'relu' ? 'selected' : ''}>ReLU</option>
                <option value="tanh" ${activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                <option value="sigmoid" ${activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                <option value="elu" ${activation === 'elu' ? 'selected' : ''}>ELU</option>
                <option value="selu" ${activation === 'selu' ? 'selected' : ''}>SELU</option>
                <option value="softplus" ${activation === 'softplus' ? 'selected' : ''}>Softplus</option>
                <option value="linear" ${activation === 'linear' ? 'selected' : ''}>Linear</option>
            </select>
        </div>
    `;
    dynamicLayersContainer.appendChild(card);
    renumberHiddenLayers(); 
}

function removeHiddenLayerUI() {
    const layers = dynamicLayersContainer.querySelectorAll('.dynamic-hidden-layer');
    if (layers.length > 0) {
        dynamicLayersContainer.removeChild(layers[layers.length - 1]);
        renumberHiddenLayers();
    } else {
        log("Nenhuma camada oculta para remover.");
    }
}

function renumberHiddenLayers() {
    const hiddenLayerCards = dynamicLayersContainer.querySelectorAll('.dynamic-hidden-layer');
    hiddenLayerCounter = hiddenLayerCards.length; 
    hiddenLayerCards.forEach((card, index) => {
        const currentLayerNum = index + 1;
        card.querySelector('.layer-header h3').textContent = `Camada Oculta ${currentLayerNum}`;
        const unitsInput = card.querySelector('.layer-units-input');
        const activationSelect = card.querySelector('.layer-activation-input');

        const oldUnitsId = unitsInput.id;
        const newUnitsId = `layer-${currentLayerNum}-units`;
        unitsInput.id = newUnitsId;
        if (unitsInput.previousElementSibling && unitsInput.previousElementSibling.tagName === 'LABEL') {
            unitsInput.previousElementSibling.setAttribute('for', newUnitsId);
        }

        const oldActivationId = activationSelect.id;
        const newActivationId = `layer-${currentLayerNum}-activation`;
        activationSelect.id = newActivationId;
        if (activationSelect.previousElementSibling && activationSelect.previousElementSibling.tagName === 'LABEL') {
            activationSelect.previousElementSibling.setAttribute('for', newActivationId);
        }
        card.dataset.layerId = currentLayerNum;
    });
}

async function saveTrainedModel() {
    if (currentBestModel) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            await currentBestModel.save(`downloads://mlp-model-${timestamp}`);
            log(`Modelo salvo: mlp-model-${timestamp}.json e mlp-model-${timestamp}.weights.bin`);
        } catch (error) {
            log(`Erro ao salvar o modelo: ${error.message}`);
            console.error(error);
        }
    } else {
        log("Nenhum modelo treinado disponível para salvar.");
    }
}

async function loadTrainedModel() {
    const jsonFile = loadModelJsonInput.files[0];
    const weightsFile = loadModelWeightsInput.files[0];

    if (!jsonFile || !weightsFile) {
        log("Por favor, selecione o arquivo .json da topologia e o arquivo .bin dos pesos.");
        return;
    }
    log("Tentando carregar modelo dos arquivos...");
    try {
        const loadedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
        if (currentBestModel) {
            currentBestModel.dispose();
        }
        currentBestModel = loadedModel;
        log("Modelo carregado com sucesso!");
        statusText.textContent = "Modelo carregado. Pronto para avaliação ou novo treinamento.";

        foldResults = [];
        plotBestModelResults(null);
        plotAllFoldsResults([]);
        displayMetrics([]);
        saveModelBtn.disabled = false; 

        log("Modelo carregado está ativo. A UI de arquitetura não reflete o modelo carregado; ela define a próxima arquitetura a ser treinada.");

    } catch (error) {
        log(`Erro ao carregar o modelo: ${error.message}`);
        console.error(error);
        statusText.textContent = `Erro ao carregar: ${error.message.substring(0,100)}...`;
    } finally {
        loadModelJsonInput.value = "";
        loadModelWeightsInput.value = "";
    }
}

function init() {
  if (typeof tf === 'undefined') {
    log('ERRO: TensorFlow.js não carregado!');
    statusText.textContent = 'ERRO: TensorFlow.js não disponível';
    trainBtn.disabled = true; saveModelBtn.disabled = true; loadModelBtn.disabled = true;
    return;
  }
  if (typeof Plotly === 'undefined') {
    log('ERRO: Plotly.js não carregado!');
    statusText.textContent = 'ERRO: Plotly.js não disponível';

  }

  loadTheme(); 
  statusText.textContent = 'Pronto para iniciar treinamento.';
  trainBtn.addEventListener('click', trainModel);

  stopTrainBtn.addEventListener('click', () => {
    log("Solicitação de interrupção do treinamento recebida...");
    stopTrainingFlag = true;
    statusText.textContent = "Tentando interromper o treinamento...";
    stopTrainBtn.disabled = true;
  });
  saveModelBtn.addEventListener('click', saveTrainedModel);
  saveModelBtn.disabled = true;
  loadModelBtn.addEventListener('click', loadTrainedModel);

  viewModeBtns.forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.mode)));
  addHiddenLayerBtn.addEventListener('click', () => addHiddenLayerUI()); 
  removeHiddenLayerBtn.addEventListener('click', removeHiddenLayerUI);

  addHiddenLayerUI(32, 'relu'); 
  addHiddenLayerUI(16, 'relu'); 

  switchView('best'); 

  plotBestModelResults(null); 
  plotAllFoldsResults([]);    

  log("Interface inicializada.");
}

window.addEventListener('resize', () => {
    const chartsToResize = [lossChartEl, accuracyChartEl, allFoldsChartEl];
    chartsToResize.forEach(chartDOMElement => {
        if (chartDOMElement && chartDOMElement.data && chartDOMElement.classList.contains('js-plotly-plot')) {
            Plotly.Plots.resize(chartDOMElement);
        }
    });
});

window.addEventListener('load', init);