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
modelFileName = 'ModelsLSTM\SERVERmodel_8-256_26-09-2023_11-11-38.mat'; 


%% THE DATASTORES RE CREATED
% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

% The datastores are created
trainingDatastore = SpectrogramDatastoreLSTM(dataDirTraining);
validationDatastore = SpectrogramDatastoreLSTM(dataDirValidation);

%dataSample = preview(trainingDatastore);
% Clean up variables
clear dataDirTraining dataDirValidation withNoGesture

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.FrameDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);


%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The total data for training-validation-tests is obtained
numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];

fprintf('\n%s\n%s\n', numTrainingSamples, numValidationSamples);

% Clean up variables
clear numTrainingSamples numValidationSamples numTestingSamples



%% NETWORK TRAINING
net = load(modelFileName).net;
% Clean up variables


%% ACCURACY FOR EACH DATASET
% The accuracy for training-validation-tests is obtained
[accTraining, flattenLabelsTraining ] = calculateAccuracy(net, trainingDatastore);
%[accValidation, flattenLabelsValidation ] = calculateAccuracy(net, validationDatastore);

% The amount of training-validation-tests data is printed
strAccTraining = ['Training accuracy: ', num2str(accTraining)];
%strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
if Shared.includeTesting
    strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
    fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
else
    fprintf('\n%s\n%s\n', strAccTraining);
end

% Clean up variables
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting







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
        size(labels{1})
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

