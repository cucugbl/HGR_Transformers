% modelCreation trains and saves the HGR CNN-LSTM model.
% In this file is stablished the neural network architecture.
% Models are generated in "ModelsLSTM/" folder.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}

% #################################################################
% #################################################################

%% SET DATASTORES PATHS
dataDirTraining = fullfile('DatastoresLSTM', 'training');
dataDirValidation = fullfile('DatastoresLSTM', 'validation');
if Shared.includeTesting
    dataDirTesting = fullfile('DatastoresLSTM', 'testing');
end

%% THE DATASTORES RE CREATED
% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

% The datastores are created
trainingDatastore = SpectrogramDatastoreLSTM(dataDirTraining);
validationDatastore = SpectrogramDatastoreLSTM(dataDirValidation);
if Shared.includeTesting
    testingDatastore = SpectrogramDatastoreLSTM(dataDirTesting);
end
%dataSample = preview(trainingDatastore);
% Clean up variables
clear dataDirTraining dataDirValidation withNoGesture

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.FrameDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);
if Shared.includeTesting
    testingDatastore = setDataAmount(testingDatastore, 1);
end

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The total data for training-validation-tests is obtained
numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];
if Shared.includeTesting
    numTestingSamples = ['Testing samples: ', num2str(testingDatastore.NumObservations)];
    fprintf('\n%s\n%s\n%s\n', numTrainingSamples, numValidationSamples, numTestingSamples);
else
    fprintf('\n%s\n%s\n', numTrainingSamples, numValidationSamples);
end
% Clean up variables
clear numTrainingSamples numValidationSamples numTestingSamples

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numClasses = trainingDatastore.NumClasses;
lgraph = setNeuralNetworkArchitecture(inputSize, numClasses);
analyzeNetwork(lgraph);
% Clean up variables
clear numClasses

%% THE OPTIONS ARE DIFINED
%gpuDevice(1);
maxEpochs = 12;%7 10
miniBatchSize = 64;%64
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',8, ... %5 8
    'ExecutionEnvironment','auto', ... %gpu
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency',floor(trainingDatastore.NumObservations/ miniBatchSize), ...
    'ValidationPatience',5, ...
    'Plots','training-progress');
% Clean up variables
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
% Clean up variables
clear options lgraph

%% ACCURACY FOR EACH DATASET
% The accuracy for training-validation-tests is obtained
%[accTraining, flattenLabelsTraining ] = calculateAccuracy(net, trainingDatastore);
[accValidation, flattenLabelsValidation ] = calculateAccuracy(net, validationDatastore);
if Shared.includeTesting
    [accTesting, flattenLabelsTesting ] = calculateAccuracy(net, testingDatastore);
end

% The amount of training-validation-tests data is printed
%strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
if Shared.includeTesting
    strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
    fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
else
    fprintf('\n%s\n%s\n', strAccTraining, strAccValidation);
end

% Clean up variables
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting

%% CONFUSION MATRIX FOR EACH DATASET
%calculateConfusionMatrix(flattenLabelsTraining, 'training');
%calculateConfusionMatrix(flattenLabelsValidation, 'validation');
if Shared.includeTesting
    calculateConfusionMatrix(flattenLabelsTesting, 'testing');
end

%% SAVE MODEL
save(['ModelsLSTM/OnlyTransformer_16_1024', datestr(now,'dd-mm-yyyy_HH-MM-ss')], 'net');

%% FUNCTION TO STABLISH THE NEURAL NETWORK ARCHITECTURE
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Create layer graph
    lgraph = layerGraph();
    
       % Add layer branches
    tempLayers = [
    sequenceInputLayer(inputSize,"Name","sequence")
    flattenLayer("Name","flatten")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = sinusoidalPositionEncodingLayer(2496,"Name","positionencode");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    selfAttentionLayer(16,512,"Name","selfattention")
    fullyConnectedLayer(numClasses,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"flatten","positionencode");
lgraph = connectLayers(lgraph,"flatten","addition/in1");
lgraph = connectLayers(lgraph,"positionencode","addition/in2");
end

%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function [accuracy, flattenLabels] = calculateAccuracy(net, datastore)
    % Configure options
    realVsPredData = cell(datastore.NumObservations, 2);
    datastore.MiniBatchSize = 1; % No padding

    % Read while the datastore has data
    totalLabels = 0;
    while hasdata(datastore)
        % Get sample
        position = datastore.CurrentFileIndex;
        data = read(datastore);
        labels = data.labelsSequences;
        sequence = data.sequences;
        % Cassify sample
        labelsPred = classify(net,sequence);
        % Save labels to flatten later and calculate accuracy
        realVsPredData(position, :) = [labels, labelsPred];
        totalLabels = totalLabels + length(labels{1,1});
    end
    
    % Allocate space for flatten labels
    flattenLabels = cell(totalLabels,2);
    idx = 0;
    % Flat labels
    for i = 1:length(realVsPredData)
        labels = realVsPredData{i, 1};
        labelsPred = realVsPredData{i, 2};
        for j = 1:length(labels)
            flattenLabels{idx+j, 1} = char(labels(1,j));
            flattenLabels{idx+j, 2} = char(labelsPred(1, j));
        end
        idx = idx + length(labels);
    end
    
    % Count labels that match with its prediction
    matches = 0;
    for i = 1:length(flattenLabels)
        if isequal(flattenLabels{i, 1}, flattenLabels{i, 2})
            matches = matches + 1;
        end
    end

    % Calculate accuracy
    accuracy = matches / length(flattenLabels);
    reset(datastore);
end

%% FUNCTION TO CALCULATE AD PLOT A CONFUSION MATRIX
function calculateConfusionMatrix(flattenLabels, datasetName)
    % Stablish clases
    classes = categorical(Shared.setNoGestureUse(true));

    % Transform labels into categorical
    realLabels = categorical(flattenLabels(:,1), Shared.setNoGestureUse(true));
    predLabels = categorical(flattenLabels(:,2), Shared.setNoGestureUse(true));
    
    % Create the confusion matrix
    confusionMatrix = confusionmat(realLabels, predLabels, 'Order', classes);
    figure('Name', ['Confusion Matrix - ' datasetName])
        matrixChart = confusionchart(confusionMatrix, classes);
        % Chart options
        matrixChart.ColumnSummary = 'column-normalized';
        matrixChart.RowSummary = 'row-normalized';
        matrixChart.Title = ['Hand gestures - ' datasetName];
        sortClasses(matrixChart,classes);
end
