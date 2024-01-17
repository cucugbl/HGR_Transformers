% modelEvaluation is used alongside the training process to evaluate training and validation samples.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}
addpath(genpath('..//CNN-LSTM'));
addpath(genpath('..//eval_HGRresponses-master'));

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG-EPN612 Dataset';
trainingDir = 'trainingJSON';


modelFileName = 'ModelsLSTM\SERVERmodel_8-256_26-09-2023_11-11-38.mat'; 

% #################################################################
% #################################################################

%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
if Shared.includeTesting
    % Divide in two datasets
    limit = length(users)- Shared.numTestUsers;
    usersTrainVal = users(1:limit, 1);
    usersTest = users(limit+1:length(users), 1);
else
    usersTrainVal = users;
end
clear dataDir trainingDir users numTestUsers limit

%% ===== JUST FOR DEBUGGING =====
%usersTrainVal = usersTrainVal(1:2);
%usersTest = usersTest(1:2);
%  ===== JUST FOR DEBUGGING =====

%% LOAD THE MODEL
model = load(modelFileName).net;
disp(modelFileName)
clear modelFile modelFileName

%% PREALLOCATE SPACE FOR RESULTS TRAINING AND VALIDATION
% Training
[classifications, recognitions, overlapings, procesingTimes] =  ... 
    deal(zeros(length(usersTrainVal), Shared.numSamplesUser));
% Validation
[classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal] = ... 
    deal(zeros(length(usersTrainVal), Shared.numSamplesUser));

%% EVALUATE EACH USER FOR TRAINING AND VALIDATION
for i = 1:length(usersTrainVal) % parfor
    fprintf('\rIteración: %d', i);
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTrainVal(i));
    
    % % Transform samples
    % transformedSamplesTraining = transformSamples(trainingSamples);
    % userResults = evaluateSamples(transformedSamplesTraining, model);
    % 
    % % Set user's training results
    % [classifications(i, :), recognitions(i, :), overlapings(i, :), procesingTimes(i, :)] = ... 
    %     deal(userResults.classifications, userResults.recognitions, ... 
    %     userResults.overlapings, userResults.procesingTimes);
    
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);

    % Set user's training results
    [classificationsVal(i, :), recognitionsVal(i, :), overlapingsVal(i, :), .... 
        procesingTimesVal(i, :)] = deal(userResults.classifications, userResults.recognitions, ... 
        userResults.overlapings, userResults.procesingTimes);
end

% % Print trainig results
% fprintf('\n\n\tTraining data results\n\n');
% resultsTrain = calculateValidationResults(classifications, recognitions, overlapings, ... 
%     procesingTimes, length(usersTrainVal));

% Print validation results
fprintf('\n\n\tValidation data results\n\n');
resultsValidation = calculateValidationResults(classificationsVal, recognitionsVal, ... 
    overlapingsVal, procesingTimesVal, length(usersTrainVal));

clear i trainingSamples validationSamples transformedSamplesValidation classifications recognitions overlapings procesingTimes classificationsVal recognitionsVal overlapingsVal procesingTimesVal

%% PREALLOCATE SPACE FOR RESULTS TESTING
if Shared.includeTesting
    % Testing - users training samples
    [classificationsTest1, recognitionsTest1, overlapingsTest1, procesingTimesTest1] =  ...
        deal(zeros(length(usersTest), Shared.numSamplesUser));
    % Testing - users validation samples
    [classificationsTest2, recognitionsTest2, overlapingsTest2, procesingTimesTest2] =  ...
        deal(zeros(length(usersTest), Shared.numSamplesUser));
end

%% EVALUATE EACH USER FOR TESTING
if Shared.includeTesting
    parfor i = 1:1 %parfor
        i
        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTest(i));

        % Transform samples
        transformedSamplesTraining = transformSamples(trainingSamples);
        userResults = evaluateSamples(transformedSamplesTraining, model);

        % Set user's training results
        classificationsTest1(i, :) = userResults.classifications; ... 
            recognitionsTest1(i, :) = userResults.recognitions; overlapingsTest1(i, :) = ... 
            userResults.overlapings; procesingTimesTest1(i, :) = userResults.procesingTimes;

        % Validation data
        transformedSamplesValidation = transformSamples(validationSamples);
        userResults = evaluateSamples(transformedSamplesValidation, model);

        % Set user's training results
        classificationsTest2(i, :) = userResults.classifications; ... 
            recognitionsTest2(i, :) = userResults.recognitions; overlapingsTest2(i, :) = ... 
            userResults.overlapings; procesingTimesTest2(i, :) = userResults.procesingTimes;
    end

    % Combine testing part (training and validation samples)
    [classificationsTest, recognitionsTest, overlapingsTest, procesingTimesTest] = ... 
        deal([classificationsTest1; classificationsTest2], [recognitionsTest1; recognitionsTest2], ... 
        [overlapingsTest1; overlapingsTest2], [procesingTimesTest1; procesingTimesTest2]);

    % Print the results
    fprintf('\n\n\tTesting data results\n\n');
    dataTest = calculateValidationResults(classificationsTest, recognitionsTest, ... 
        overlapingsTest, procesingTimesTest, length(usersTest));
end
% Clear variables
clear i trainingSamples validationSamples transformedSamplesValidation classificationsTest1 recognitionsTest1 overlapingsTest1 procesingTimesTest1 classificationsTest2 recognitionsTest2 overlapingsTest2 procesingTimesTest2n classificationsTest recognitionsTest overlapingsTest procesingTimesTest

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = transformSamples(samples)    
    % Get sample keys
    samplesKeys = fieldnames(samples);
    
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys), 3);
    
    for i = 1:length(samplesKeys)       
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        
        % Adding the transformed data
        transformedSamples{i,1} = signal;
        transformedSamples{i,2} = gestureName;
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            transformedSamples{i,3} = transpose(groundTruth);
        end
    end
end

%% FUCTION TO GENERATE THE DATA is the same that spectrogramdatasetgeneration
function transformedSamples = generateData(samples)

    % Number of noGesture samples to discard them
    noGesturePerUser = Shared.numGestureRepetitions;
    
    % Allocate space for the results
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(length(samplesKeys)- noGesturePerUser, 3);
    
    % For each gesture sample
    for i = noGesturePerUser + 1:length(samplesKeys)
        
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        
        % Generate spectrograms
        [data, newGroundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName);
        
        % Save the transformed data
        transformedSamples{i - noGesturePerUser, 1} = data;
        transformedSamples{i - noGesturePerUser, 2} = gestureName;
        transformedSamples{i - noGesturePerUser, 3} = transpose(newGroundTruth); % Can have filling
    end
end


%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(numObservations)
    % Allocate space to save the results
    clasifications = zeros(numObservations, 1);
    recognitions = -1*ones(numObservations, 1);
    overlapings = -1*ones(numObservations, 1);
    procesingTimes = zeros(numObservations, 1);
end

%% FUNCTION TO EVALUATE SAMPLE FRAMES
function [labels, timestamps, processingTimes] = evaluateSampleFrames(signal, groundTruth, net)

    % Fill before frame classification
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
        
        % Get a nogesture portion of the sample to use as filling
        if groundTruth
            noGestureInSignal = signal(~groundTruth, :);
            filling = noGestureInSignal(1: floor(Shared.FRAME_WINDOW / 2), :);
        else
            filling = signal(1: floor(Shared.FRAME_WINDOW / 2), :);
        end
        % Combine the sample with the filling
        signal = [signal; filling];
        
    end
    
    % Allocate space for the results
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    
    % Preallocate space for the spectrograms
    labels = cell(1, numWindows);
    timestamps = zeros(1,numWindows);
    processingTimes = zeros(1,numWindows);
    
    % Reset the network for each sample
    net = resetState(net);
    sequence = zeros(13, 24, 8, 16);
    % Creating frames
    for i = 1:numWindows
        
        % Start timer
        timer = tic;
        
        % Get signal data to create a frame
        traslation = ((i-1)* Shared.WINDOW_STEP_LSTM);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        
        % Calculate timestamp
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        
        % Get signal of the window
        frameSignal = signal(inicio:finish, :);
        frameSignal = Shared.preprocessSignal(frameSignal);
        
        % Get Spectrogram of the window (frame)
        spectrograms = Shared.generateSpectrograms(frameSignal);
        sequence(:, :, :, i) = spectrograms;
        % Classify the frame
        %[predicction, predictedScores] = classify(net,spectrograms);
        
        % Check if the prediction is over the frame classification threshold
        % if max(predictedScores) < Shared.FRAME_CLASS_THRESHOLD
        %     label = 'noGesture';
        % else
        %     label = char(predicction);
        % end
        
        % Stop timer
        processingTime = toc(timer);
        
        % Save sample results
        %labels{1, i} =  label; 
        timestamps(1, i) = timestamp; 
        processingTimes(1, i) = processingTime;
    end
    predicctionS = classify(net,sequence);      

    cell_array = cell(size(timestamps));
    % Convierte el array en un cell array
    for k = 1:numel(timestamps)
         cell_array{k} = char(predicctionS(k));
    end
    
    labels=cell_array;
   
end

%% FUNCTION TO EVALUETE SAMPLES OF A USER
function userResults = evaluateSamples(samples, model)
    
    % Preallocate space for results
    [classifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(length(samples));
    
    % For each sample of a user
    for i = 1:length(samples)
        
        % Get sample data
        emg = samples{i, 1};
        gesture = samples{i, 2};
        groundTruth = samples{i, 3};
        
        % Prepare repetition information
        if ~isequal(gesture,'noGesture')
            repInfo.groundTruth = logical(groundTruth);
        end
        repInfo.gestureName = categorical({gesture}, Shared.setNoGestureUse(true));
        
        % Evaluate a sample with slidding window
        [labels, timestamps, processingTimes] = evaluateSampleFrames(emg, groundTruth, model);
        
        % Set a class for the sample
        class = Shared.classifyPredictions(labels);


      
        
        % Postprocess the sample (labels)
        %%%%%%labels = Shared.postprocessSample(labels, char(class));
        % Transform to categorical
        %labels = categorical(labels, Shared.setNoGestureUse(true));
             
llll=labels;

        posprocesamiento = 1;
        if posprocesamiento==1
            if ~isequal(gesture,'noGesture')
                for indexL = 1:length(labels)
                    % si es una etique ta diferente a no gesto, entra a la siguiente
                    % sentencia
                    if ~isequal(labels{indexL}, 'noGesture') && ~isequal(labels{indexL}, class)
                       
                         % Verifica si es el primer elemento o el último, o si
                         % tiene 'nogesture' a los dos lados al mismo tiempo
                        if indexL == 1 || indexL == length(labels) || (strcmp(labels{indexL-1}, 'noGesture')) || indexL == length(labels)
                            labels{indexL} = 'noGesture';
                        else 
                            labels{indexL} = class{1};  
                            
                        end
                    elseif indexL > 3 && indexL < length(labels)-2 && isequal(labels{indexL}, class) 
                        
                        if (~strcmp(labels{indexL+1}, class) && strcmp(labels{indexL+2}, class)) || (~strcmp(labels{indexL+1}, class) && strcmp(labels{indexL+3}, class))
                            
                            labels{indexL+1} = class{1};  
                        end
                    elseif indexL > length(labels)-2 &&  (strcmp(labels{indexL-1}, 'noGesture'))
                            labels{indexL} = 'noGesture';  
                    
                    elseif indexL < 3  &&  (strcmp(labels{indexL+1}, 'noGesture'))
                            labels{indexL} = 'noGesture'; 
                    end
                end        
            end
        
        end
        
        % Transform to categorical
        labels = categorical(labels, Shared.setNoGestureUse(true));
    
        % Prepare response
        response = struct('vectorOfLabels', labels, 'vectorOfTimePoints', timestamps, ... 
            'vectorOfProcessingTimes', processingTimes, 'class', class);
        
        % Send to validation toolbox



        result = evalRecognition(repInfo, response);
        
        % Save results

        classifications(i) = result.classResult;
        if ~isequal(gesture,'noGesture')
            recognitions(i) = result.recogResult;
            overlapings(i) = result.overlappingFactor;
        end
        procesingTimes(i) = mean(processingTimes); %time (frame)

        % Set User Results
        userResults = struct('classifications', classifications,  'recognitions', ... 
            recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (MEAN USERS)
function [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
            calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers)

    % Allocate space for results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
        deal(zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1));
    
    % Calculate results per user
    for i = 1:numUsers
        classificationPerUser(i, 1) = sum(classifications(i, :) == 1) / length(classifications(i, :));
        % NoGesture omitted it has value = -1 
        recognitionPerUser(i , 1) = sum(recognitions(i, :) == 1) / ... 
            sum(recognitions(i, :) == 1 | recognitions(i, :) == 0);
        overlapingsUser = overlapings(i, :);
        overlapingPerUser(i, 1) = mean(overlapingsUser(overlapingsUser ~= -1),'omitnan');
        processingTimePerUser(i, 1) = mean(procesingTimes(i, :));
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (GLOBAL)
function [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers)
    
    % Calculate accuracies 
    accClasification = sum(all.classifications==1) / length(all.classifications);
    % NoGesture omitted it has value = -1 
    accRecognition = sum(all.recognitions==1) / sum(all.recognitions == 1 | all.recognitions == 0); 
    avgOverlapingFactor = mean(all.overlapings(all.overlapings ~= -1), 'omitnan');
    avgProcesingTime = mean(all.procesingTimes);
    
    % Set results
    globalResps = struct('accClasification', accClasification, 'accRecognition', accRecognition, ... 
        'avgOverlapingFactor', avgOverlapingFactor, 'avgProcesingTime', avgProcesingTime);
    
    % Stract data per user
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = deal( ... 
        perUser.classifications, perUser.recognitions, perUser.overlapings, perUser.procesingTimes);
    [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    
    % Calculate standard deviations regarding users means
    for i = 1:numUsers
        stdClassification = stdClassification + (classificationPerUser(i,1) - accClasification) ^ 2;
        stdRecognition = stdRecognition + (recognitionPerUser(i, 1) - accRecognition) ^ 2;
        stdOverlaping = stdOverlaping + (overlapingPerUser(i, 1) - avgOverlapingFactor) ^ 2;
        stdProcessingTime = stdProcessingTime + (processingTimePerUser(i, 1) - avgProcesingTime) ^ 2;
    end
    
    % Check number of users
    if numUsers > 1
         [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal( ... 
             stdClassification / (numUsers - 1), stdRecognition / (numUsers - 1), ... 
             stdOverlaping / (numUsers - 1), stdProcessingTime / (numUsers - 1));
    else 
        [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    end
    
    % Set standard deviations
    globalStds = struct('stdClassification', stdClassification, 'stdRecognition', stdRecognition, ... 
        'stdOverlaping', stdOverlaping, 'stdProcessingTime', stdProcessingTime);
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE
function results = calculateValidationResults(classifications, recognitions, overlapings, procesingTimes, numUsers)
    
    % Calculate results using the mean values of users results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
    calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers);
    
    % Print results using mean values
    disp('Results (mean of user results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        mean(classificationPerUser), std(classificationPerUser));
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        mean(recognitionPerUser), std(recognitionPerUser));
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        mean(overlapingPerUser), std(overlapingPerUser));
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        mean(processingTimePerUser), std(processingTimePerUser));
    
    % Flatten samples
    [classifications, recognitions, overlapings, procesingTimes] = ... 
    deal(classifications(:), recognitions(:), overlapings(:), procesingTimes(:));
    
    % Organize data in structs
    all = struct('classifications', classifications, 'recognitions', recognitions, ... 
        'overlapings', overlapings, 'procesingTimes', procesingTimes);
    perUser =  struct('classifications', classificationPerUser, 'recognitions', recognitionPerUser, ... 
        'overlapings', overlapingPerUser, 'procesingTimes', processingTimePerUser);
    
    % Calculate results using a global mean
    [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers);
    
    % Print results using global values
    disp('Results (Global results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        globalResps.accClasification, globalStds.stdClassification);
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        globalResps.accRecognition, globalStds.stdRecognition);
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        globalResps.avgOverlapingFactor, globalStds.stdOverlaping);
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        globalResps.avgProcesingTime, globalStds.stdProcessingTime);
    
    % Set results
    results = struct('clasification',  globalResps.accClasification, 'recognition', ... 
        globalResps.accRecognition, 'overlapingFactor', globalResps.avgOverlapingFactor, ... 
        'procesingTime', globalResps.avgProcesingTime);
end



%% FUNCTION TO GENERATE FRAMES
function [data, groundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName)

    % Fill before frame classification
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
        
        % Get a nogesture portion of the sample to use as filling
        noGestureInSignal = signal(~groundTruth, :);
        filling = noGestureInSignal(1: floor(Shared.FRAME_WINDOW / 2), :);

        % Combine the sample with the filling
        signal = [signal; filling];
        groundTruth = [groundTruth; zeros(floor(Shared.FRAME_WINDOW / 2), 1)];
    end

    % Allocate space for the results
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    data = cell(numWindows, 3);
    data(:,2) = {'noGesture'};
    isIncluded = false(numWindows, 1);
    
    % Creating frames
    for i = 1:numWindows
        
        % Get signal data to create a frame
        traslation = ((i-1)* Shared.WINDOW_STEP_LSTM);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameGroundTruth = groundTruth(inicio:finish);
        totalOnes = sum(frameGroundTruth == 1);
        
         % Get Spectrogram of the window
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        
        % Set data
        data{i,1} = spectrograms; % datum
        data{i,3} = timestamp; % time
        
        % Check the thresahold to consider gesture
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE_LSTM
            isIncluded(i,1) = true;
            data{i,2} = gestureName;
        end
    end
    
    % Include nogestures in the sequence
    if isequal(Shared.NOGESTURE_FILL, 'all')
        
        isIncluded(:,1) = true;
        
    elseif isequal(Shared.NOGESTURE_FILL, 'some')
        
        first = find(isIncluded, true, 'first');
        last = find(isIncluded, true, 'last');
        
        for i = 1:Shared.NOGESTURE_IN_SEQUENCE
            % Include some from left
           if first - i >= 1
                isIncluded(first-i, 1) = true;
           end
           % Include some from right
           if last + i <= numWindows
                isIncluded(last + i, 1) = true;
           end
        end
        
    end
    
    % Filer results    
    data = data(isIncluded,:);
            
end

