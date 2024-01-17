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
modelFileName = 'ModelsLSTM\model_02-08_5M_91068_8_64.mat'; 

%% SET DATASTORES PATHS
dataDirTraining = fullfile('DatastoresLSTM', 'training');
dataDirValidation = fullfile('DatastoresLSTM', 'validation');
if Shared.includeTesting
    dataDirTesting = fullfile('DatastoresLSTM', 'testing');
end

%% THE DATASTORES ARE CREATED
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


%% NETWORK EVALUATION

net = load(modelFileName).net;

fprintf('Modelo evaluation: %s', modelFileName)
clear modelFile modelFileName

%% ACCURACY FOR EACH DATASET
% The accuracy for training-validation-tests is obtained
[accTraining, flattenLabelsTraining,evalRecognitionTranining ] = calculateAccuracy(net, trainingDatastore);
[accValidation, flattenLabelsValidation,evalRecognitionValidation ] = calculateAccuracy(net, validationDatastore);
if Shared.includeTesting
    [accTesting, flattenLabelsTesting ] = calculateAccuracy(net, testingDatastore);
end

% The amount of training-validation-tests data is printed
strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strRecTraining = ['Recognition: ', num2str(evalRecognitionTranining.recognitions),...
    '| Overlaping: ', num2str(evalRecognitionTranining.overlapings)];

strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
strRecValidation = ['Recognition: ', num2str(evalRecognitionValidation.recognitions),...
    '| Overlaping: ', num2str(evalRecognitionValidation.overlapings)];

if Shared.includeTesting
    strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
    fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
else
    fprintf('\n%s\n%s\n%s\n%s\n', strAccTraining,strRecValidation, strAccValidation,strRecTraining);
end

% Clean up variables
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting strRecTraining strRecValidation


%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function [accuracy, flattenLabels,TResults] = calculateAccuracy(net, datastore)
    % Configure options
    realVsPredData = cell(datastore.NumObservations, 2);
    datastore.MiniBatchSize = 1; % No padding
    
    recognitions = -1*ones(datastore.NumObservations, 1);
    overlapings = -1*ones(datastore.NumObservations, 1);
    procesingTimes = -1*ones(datastore.NumObservations, 1);

    % Read while the datastore has data
    totalLabels = 0;
    %while hasdata(datastore)
    for i= 1:datastore.NumObservations
    % Get sample
        position = datastore.CurrentFileIndex;
        [data,info] = read(datastore);
       
        labelsTarget = data.labelsSequences;
        sequence = data.sequences;

        % Start timer
        timer = tic;
        % Cassify sample
        labelsPred = classify(net,sequence);
        processingTime = toc(timer);
        processingTimes = processingTime;
        %%Recognition 
        if ~isequal(info.responses,'noGesture')
            repInfo.groundTruth = info.groundTruths;
        end
        repInfo.gestureName = info.responses;
        % Set a class for the sample
        labels=cellstr(labelsPred{1});
        vectorOfTimePoints=cell2mat(info.timestamps);
        

        class = categorical(Shared.classifyPredictions(labels));

        labels = Shared.postprocessSample(labels, char(class));
        % Prepare response
        response = struct('vectorOfLabels', labels, 'vectorOfTimePoints',vectorOfTimePoints , ... 
            'vectorOfProcessingTimes', processingTimes,'class', class);

        result = evalRecognition(repInfo, response);
        
        % Save results
        if ~isequal(info.responses,'noGesture')
            recognitions(i) = result.recogResult;
            overlapings(i) = result.overlappingFactor;
        end
        procesingTimes(i) = mean(processingTimes); %time (frame)
        
        %%Recognition 

        % Save labels to flatten later and calculate accuracy
        realVsPredData(position, :) = [labelsTarget, labelsPred];
        totalLabels = totalLabels + length(labelsTarget{1,1});
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

    % Set User Results

    recognitions = sum(recognitions(:, 1) == 1) / ... 
            sum(recognitions(:, 1) == 1 | recognitions(:, 1) == 0);
    overlapingsAux = overlapings(:, 1);
    overlapingMean = mean(overlapingsAux(overlapings ~= -1),'omitnan');
    procesingAux = procesingTimes(:, 1);
    processingMean = mean(procesingAux(procesingTimes~= -1));
    TResults = struct('recognitions', ... 
            recognitions, 'overlapings', overlapingMean, 'procesingTimes', processingMean);


    % Calculate accuracy
    accuracy = matches / length(flattenLabels);
    reset(datastore);
end

