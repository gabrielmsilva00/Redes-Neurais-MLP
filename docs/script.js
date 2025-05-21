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
  const layout = getCurrentPlotlyThemeLayout(themeName);
  const charts = ['loss-chart', 'accuracy-chart', 'all-folds-chart'];
  charts.forEach(chartId => {
    const chartElem = document.getElementById(chartId);
    if (chartElem && chartElem.classList.contains('js-plotly-plot')) {
      Plotly.relayout(chartElem, layout);
    }
  });
}

function getCurrentPlotlyThemeLayout(themeName = null) {
    const currentTheme = themeName || localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    let bgColor, gridColor, textColor, titleColor;
    switch (currentTheme) {
        case 'light': bgColor = '#ffffff'; gridColor = '#e1e1e1'; textColor = '#000000'; titleColor = '#000000'; break;
        case 'dark': bgColor = '#1e1e1e'; gridColor = '#333333'; textColor = '#ffffff'; titleColor = '#ffffff'; break;
        case 'black': bgColor = '#000000'; gridColor = '#222222'; textColor = '#ffffff'; titleColor = '#ffffff'; break;
        default: bgColor = '#1e1e1e'; gridColor = '#333333'; textColor = '#ffffff'; titleColor = '#ffffff';
    }
    return {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: { color: textColor },
        titlefont: { color: titleColor },
        xaxis: { gridcolor: gridColor, zerolinecolor: gridColor, linecolor: gridColor, tickfont: {color: textColor}, titlefont: {color: textColor} },
        yaxis: { gridcolor: gridColor, zerolinecolor: gridColor, linecolor: gridColor, tickfont: {color: textColor}, titlefont: {color: textColor} }
    };
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
const thresholdsMinInput = document.getElementById('thresholds-min');
const thresholdsMaxInput = document.getElementById('thresholds-max');

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
const lossChart = document.getElementById('loss-chart');
const accuracyChart = document.getElementById('accuracy-chart');
const allFoldsChart = document.getElementById('all-folds-chart');
const metricsElement = document.getElementById('metrics');
const viewModeBtns = document.querySelectorAll('.view-mode-btn');

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
  logMessages.push(formattedMessage);
  if (logMessages.length > 200) logMessages.shift();
  logElement.innerHTML = logMessages.join('<br>');
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
                if (xTrainParts.length === 0) { // Should only happen if k=1 and val takes all data, or foldSize is too large
                    xTrain = xShuffled.gather(tf.tensor1d([], 'int32')); // Empty tensor
                    yTrain = yShuffled.gather(tf.tensor1d([], 'int32')); // Empty tensor
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
    this.getStopSignal = getStopSignal; // Função para verificar a flag de interrupção
  }
  async onEpochEnd(epoch, logs) { // Tornar async para usar await tf.nextFrame()
    let trainingShouldStop = false;
    if (this.getStopSignal && this.getStopSignal()) {
      log(`Sinal de interrupção recebido na Época ${epoch + 1} do ${this.foldIdentifier}. Model.stopTraining será definido como true.`);
      if (this.model) { // this.model é definido internamente pelo TensorFlow.js
        this.model.stopTraining = true; // Sinaliza ao TFJS para parar após esta época
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
    
    // Ceder controle para o navegador para atualizar a UI e processar eventos
    await tf.nextFrame(); 
  }
}

async function trainModel() {
  if (isTraining) { log("Treinamento já está em andamento."); return; }
  isTraining = true;
  stopTrainingFlag = false; // Resetar a flag no início de um novo treinamento
  // Atualizar UI dos botões
  trainBtn.textContent = "Treinando...";
  trainBtn.disabled = true;
  document.getElementById('stop-train-btn').style.display = 'inline-block';
  document.getElementById('stop-train-btn').disabled = false;
  saveModelBtn.disabled = true;
  loadModelBtn.disabled = true;
  log("Iniciando processo de treinamento...");
  updateProgress(0, "Preparando dados e modelo...");
  await tf.nextFrame(); // Permitir atualização inicial da UI
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
    tf.setBackend('cpu'); 
    if (globalData.x) tf.dispose(globalData.x);
    if (globalData.y) tf.dispose(globalData.y);
    globalData = generateData(nSamples, nFeatures);
    log(`Dados gerados: ${nSamples} amostras, ${nFeatures} features.`);
    await tf.nextFrame();
    const folds = getKFolds(globalData.x, globalData.y, nSplits, testSize);
    log(`Dados divididos em ${folds.length} folds.`);
    await tf.nextFrame();
    if (currentBestModel) { currentBestModel.dispose(); currentBestModel = null;}
    foldResults = [];
    let bestFoldIndex = -1;
    let bestValAccuracy = -1;
    for (let i = 0; i < folds.length; i++) {
      if (stopTrainingFlag) {
          log(`Treinamento interrompido pelo usuário antes do Fold ${i + 1}.`);
          break; // Sair do loop de folds
      }
      updateProgress(i / folds.length, `Preparando Fold ${i + 1}/${folds.length}...`);
      await tf.nextFrame(); // Ceder controle entre folds
      
      log(`--- Fold ${i + 1} ---`);
      
      const { train, val } = folds[i];
      if (!train.x || !train.y || !val.x || !val.y || train.x.shape[0] === 0 || val.x.shape[0] === 0) {
          log(`Fold ${i+1} ignorado devido a dados de treino/validação insuficientes. Train X: ${train.x ? train.x.shape : 'null'}, Val X: ${val.x ? val.x.shape : 'null'}`);
          continue;
      }
      const model = createModel(nFeatures, layerDefs, optimizerName, learningRateVal);
      if (!model) {
          log(`Falha ao criar modelo para o Fold ${i+1}.`);
          continue;
      }
      
      // Passar uma função getter para a flag de interrupção
      const customLogger = new CustomLogger(`Fold ${i + 1}`, maxEpochs, i, folds.length, () => stopTrainingFlag);
      let history;
      try {
        history = await model.fit(train.x, train.y, {
          epochs: maxEpochs,
          batchSize: batchSize,
          validationData: [val.x, val.y],
          classWeight: {0: classWeight0, 1: classWeight1},
          callbacks: [
            tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: patience}),
            customLogger
          ]
        });
      } catch (fitError) {
          log(`Erro durante model.fit() para o Fold ${i+1}: ${fitError.message}`);
          if (model) model.dispose();
          tf.dispose([train.x, train.y, val.x, val.y]);
          continue; // Pular para o próximo fold
      }
      // Verificar se o treinamento foi interrompido durante este fold
      if (stopTrainingFlag && model.stopTraining) {
          log(`Fold ${i + 1} interrompido durante o treinamento.`);
      }
      
      const finalValMetrics = await model.evaluate(val.x, val.y, {batchSize: batchSize});
      const valLoss = finalValMetrics[0].dataSync()[0];
      const valAcc = finalValMetrics[1].dataSync()[0];
      tf.dispose(finalValMetrics); // Descartar tensores de evaluate
      log(`Fold ${i + 1} - Validação Final: Perda = ${valLoss.toFixed(4)}, Acurácia = ${(valAcc * 100).toFixed(2)}%`);
      
      // Adicionar resultados apenas se o histórico existir (não interrompido antes da primeira época)
      if (history && history.history) {
          foldResults.push({ 
              loss: history.history.loss || [], 
              accuracy: history.history.acc || [], 
              val_loss: history.history.val_loss || [], 
              val_accuracy: history.history.val_acc || [], 
              final_val_loss: valLoss, 
              final_val_accuracy: valAcc 
          });
      } else if (!stopTrainingFlag) { // Se não foi interrompido, mas não há histórico, é estranho
          log(`AVISO: Fold ${i+1} completou model.fit() mas não produziu histórico válido.`);
      }
      
      if (valAcc > bestValAccuracy && !stopTrainingFlag) { // Não atualizar o melhor modelo se o treinamento foi interrompido
        bestValAccuracy = valAcc;
        bestFoldIndex = i;
        if (currentBestModel) currentBestModel.dispose();
        currentBestModel = model; // Manter este modelo
      } else {
        model.dispose(); // Descartar o modelo treinado neste fold
      }
      tf.dispose([train.x, train.y, val.x, val.y]); // Descartar dados do fold
    } // Fim do loop de folds
    
    if (stopTrainingFlag) {
        log("Processo de treinamento interrompido pelo usuário.");
        updateProgress( (foldResults.length / (folds.length || 1) ), "Treinamento interrompido.");
    } else if (foldResults.length > 0) {
        log("Treinamento concluído para todos os folds processados.");
        updateProgress(1, "Treinamento concluído.");
    } else {
        log("Nenhum fold foi treinado com sucesso ou o treinamento foi interrompido muito cedo.");
        updateProgress(0, "Treinamento finalizado sem resultados de folds.");
    }
    
    // Plotar resultados mesmo se interrompido, com o que tiver sido coletado
    if (currentBestModel && bestFoldIndex !== -1 && foldResults[bestFoldIndex]) {
        log(`Melhor modelo (Fold ${bestFoldIndex + 1}) com acurácia de validação de ${(bestValAccuracy * 100).toFixed(2)}% mantido.`);
        plotBestModelResults(foldResults[bestFoldIndex]);
    } else if (foldResults.length > 0 && foldResults[0]){
        log("Exibindo resultados do primeiro fold válido ou do último fold parcialmente treinado.");
        plotBestModelResults(foldResults[0]);
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
    trainBtn.textContent = "Iniciar Treinamento";
    trainBtn.disabled = false;
    document.getElementById('stop-train-btn').style.display = 'none';
    document.getElementById('stop-train-btn').disabled = true;
    saveModelBtn.disabled = !currentBestModel;
    loadModelBtn.disabled = false;
    // stopTrainingFlag será resetado no próximo clique de 'trainBtn'
    log("Rotina de finalização do treinamento executada.");
    await tf.nextFrame(); // Garantir que a UI seja atualizada após tudo
  }
}

function plotBestModelResults(bestFoldData) {
  if (!bestFoldData || !bestFoldData.loss || !bestFoldData.val_loss) { 
    Plotly.purge(lossChart); Plotly.purge(accuracyChart);
    log("Dados insuficientes para plotar o melhor modelo."); return; 
  }
  const epochs = Array.from({ length: bestFoldData.loss.length }, (_, i) => i + 1);
  
  const lossTraceTrain = { x: epochs, y: bestFoldData.loss, mode: 'lines', name: 'Perda (Treino)', line: {color: 'blue'} };
  const lossTraceVal = { x: epochs, y: bestFoldData.val_loss, mode: 'lines', name: 'Perda (Validação)', line: {color: 'orange'} };
  const accuracyTraceTrain = { x: epochs, y: bestFoldData.accuracy.map(a => a*100), mode: 'lines', name: 'Acurácia (Treino)', line: {color: 'green'} };
  const accuracyTraceVal = { x: epochs, y: bestFoldData.val_accuracy.map(a => a*100), mode: 'lines', name: 'Acurácia (Validação)', line: {color: 'red'} };
  
  const layout = getCurrentPlotlyThemeLayout();
  Plotly.newPlot(lossChart, [lossTraceTrain, lossTraceVal], { ...layout, title: 'Curvas de Perda (Melhor Fold)', yaxis: {...layout.yaxis, title: 'Perda'} , xaxis: {...layout.xaxis, title: 'Época'} });
  Plotly.newPlot(accuracyChart, [accuracyTraceTrain, accuracyTraceVal], { ...layout, title: 'Curvas de Acurácia (Melhor Fold)', yaxis: {...layout.yaxis, title: 'Acurácia (%)'} , xaxis: {...layout.xaxis, title: 'Época'} });
  switchView('best');
}

function plotAllFoldsResults(results) {
  if (!results || results.length === 0) { Plotly.purge(allFoldsChart); return; }
  const tracesLoss = [];
  
  results.forEach((foldData, index) => {
    if (foldData && foldData.val_loss && foldData.val_loss.length > 0) {
        const epochs = Array.from({ length: foldData.val_loss.length }, (_, i) => i + 1);
        tracesLoss.push({ x: epochs, y: foldData.val_loss, mode: 'lines', name: `Fold ${index + 1} Perda Val.` });
    }
  });
  
  const layout = getCurrentPlotlyThemeLayout();
  if(tracesLoss.length > 0) {
    Plotly.newPlot(allFoldsChart, tracesLoss, { ...layout, title: 'Perda de Validação (Todos Folds)', yaxis: {...layout.yaxis, title: 'Perda'} , xaxis: {...layout.xaxis, title: 'Época'} });
  } else {
    Plotly.purge(allFoldsChart);
  }
}

function displayMetrics(results) {
  if (!results || results.length === 0) {
    metricsElement.innerHTML = '<p>Nenhuma métrica disponível. Execute o treinamento.</p>';
    return;
  }
  let metricsHTML = `<h3>Métricas Finais de Validação por Fold</h3><table><thead><tr><th>Fold</th><th>Perda</th><th>Acurácia</th></tr></thead><tbody>`;
  let totalLoss = 0, totalAcc = 0, validFolds = 0;
  let bestFoldOverallAccuracy = -1, bestFoldOverallIndex = -1;

  results.forEach((fold, index) => {
    if (fold && typeof fold.final_val_loss !== 'undefined' && typeof fold.final_val_accuracy !== 'undefined') {
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
    if (fold && typeof fold.final_val_loss !== 'undefined' && typeof fold.final_val_accuracy !== 'undefined') {
      const loss = fold.final_val_loss;
      const acc = fold.final_val_accuracy * 100;
      const isBest = index === bestFoldOverallIndex;
      metricsHTML += `<tr class="${isBest ? 'highlight' : ''}"><td>${index + 1}</td><td>${loss.toFixed(4)}</td><td>${acc.toFixed(2)}%</td></tr>`;
    }
  });
  
  if (validFolds > 0) {
    const avgLoss = totalLoss / validFolds;
    const avgAcc = (totalAcc / validFolds) * 100;
    metricsHTML += `<tr style="font-weight: bold;"><td>Média</td><td>${avgLoss.toFixed(4)}</td><td>${avgAcc.toFixed(2)}%</td></tr>`;
  } else {
    metricsHTML += `<tr><td colspan="3">Nenhum fold completou com sucesso.</td></tr>`;
  }
  metricsHTML += `</tbody></table>`;
  metricsElement.innerHTML = metricsHTML;
}

function switchView(mode) {
  document.getElementById('best-model-view').style.display = (mode === 'best' ? 'block' : 'none');
  document.getElementById('all-folds-view').style.display = (mode === 'all' ? 'block' : 'none');
  viewModeBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
}

function addHiddenLayerUI(units = 16, activation = 'relu') {
    hiddenLayerCounter++;
    const layerId = hiddenLayerCounter;
    const card = document.createElement('div');
    card.classList.add('layer-card', 'elevation-1', 'dynamic-hidden-layer');
    card.dataset.layerId = layerId;
    card.innerHTML = `
        <div class="layer-header">
            <h3>Camada Oculta ${layerId}</h3>
        </div>
        <div class="form-group">
            <label for="layer-${layerId}-units">Unidades:</label>
            <input type="number" id="layer-${layerId}-units" value="${units}" min="1" max="512" class="md-input layer-units-input">
        </div>
        <div class="form-group">
            <label for="layer-${layerId}-activation">Ativação:</label>
            <select id="layer-${layerId}-activation" class="md-input layer-activation-input">
                <option value="relu" ${activation === 'relu' ? 'selected' : ''}>ReLU</option>
                <option value="tanh" ${activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                <option value="sigmoid" ${activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                <option value="elu" ${activation === 'elu' ? 'selected' : ''}>ELU</option>
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
        if (dynamicLayersContainer.children.length === 0) hiddenLayerCounter = 0;
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
        unitsInput.id = `layer-${currentLayerNum}-units`;
        unitsInput.previousElementSibling.setAttribute('for', `layer-${currentLayerNum}-units`);
        activationSelect.id = `layer-${currentLayerNum}-activation`;
        activationSelect.previousElementSibling.setAttribute('for', `layer-${currentLayerNum}-activation`);
        card.dataset.layerId = currentLayerNum;
    });
}

async function saveTrainedModel() {
    if (currentBestModel) {
        try {
            await currentBestModel.save('downloads://trained_mlp_model');
            log("Modelo salvo com sucesso. Verifique seus downloads (trained_mlp_model.json e trained_mlp_model.weights.bin).");
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
        statusText.textContent = `Erro ao carregar: ${error.message}`;
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
  
  loadTheme();
  statusText.textContent = 'Pronto para iniciar treinamento.';
  trainBtn.addEventListener('click', trainModel);
  
  const stopTrainBtn = document.getElementById('stop-train-btn');
  stopTrainBtn.addEventListener('click', () => {
    log("Solicitação de interrupção do treinamento recebida...");
    stopTrainingFlag = true;
    statusText.textContent = "Tentando interromper o treinamento...";
    stopTrainBtn.disabled = true; // Prevenir múltiplos cliques
  });
  saveModelBtn.addEventListener('click', saveTrainedModel);
  saveModelBtn.disabled = true;
  loadModelBtn.addEventListener('click', loadTrainedModel);
  
  viewModeBtns.forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.mode)));
  addHiddenLayerBtn.addEventListener('click', () => addHiddenLayerUI());
  removeHiddenLayerBtn.addEventListener('click', removeHiddenLayerUI);
  addHiddenLayerUI(12, 'relu');
  addHiddenLayerUI(8, 'relu');
  switchView('best');
  updatePlotlyTheme();
  log("Interface inicializada.");
}

window.addEventListener('load', init);