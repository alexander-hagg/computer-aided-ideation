function predMap = sail(map,fitnessFunction,p,d,varargin)
%SAIL - Surrogate Assisted Illumination Algorithm
% Main run script of SAIL algorithm
%
% Author: Adam Gaier, Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: adam.gaier@h-brs.de, alexander.hagg@h-brs.de
% Nov 2016; Last revision: 23-Aug-2019

p.infill = infillParamSet;

p.predMapResolution = p.featureResolution;
p.featureResolution = p.infill.featureResolution;

observation = reshape(map.genes,[],d.dof);
valid = all(~isnan(observation)');
observation = observation(valid,:);
nSamples = size(observation,1);
p.numInitSamples = nSamples; %Reduce to valid solutions ... bit hacky
fitness = reshape(map.fitness,numel(valid),[]);
fitness = fitness(valid,:);

p.infill.modelParams = paramsGP(size(observation,2));
while nSamples <= p.infill.nTotalSamples
    %% 1 - Create Surrogate and Acquisition Function 
    % Surrogate models are created from all evaluated samples, and these
    % models are used to produce an acquisition function.
    % Only retrain model parameters every 'p.trainingMod' iterations
    p.infill.model.functionEvals = 0;
    if (nSamples==p.numInitSamples || mod(nSamples,p.infill.trainingMod*p.infill.nAdditionalSamples))
        p.infill.model.functionEvals = 100;
    end
    model = trainGP(observation,fitness,p.infill.modelParams);
    
    % Save found model parameters and new acquisition function
    %for iModel=1:size(value,2)
    %    if strcmp(d.paramsAcq{iModel}.name,'GP'); d.paramsAcq{iModel}.hyp = model{iModel}.hyp; end
    %end
    acqFunction = createAcquisitionFcn(fitnessFunction,model,d);
    
    % After final model is created no more infill is necessary
    if nSamples == p.infill.nTotalSamples; break; end
    
    %% 2 - Illuminate Acquisition Map
    % A map is constructed using the evaluated samples which are evaluated
    % with the acquisition function and placed in the map as the initial
    % population. The observed samples are the seed population of the
    % 'acquisition map' which is then created by optimizing the acquisition
    % function with MAP-Elites.
    
    % Evaluate data set with acquisition function
    try
        [fitness,values,phenotypes] = acqFunction(observation);
    catch exception
        disp(exception.identifier);
    end
    
    %map = updateMap(replaced,replacement,map,fitness,initSamples,values,features,p.extraMapValues);
    % Place Best Samples in Map with Acquisition Fitness
    obsMap = createMap(d,p);
    [replaced, replacement, features] = nicheCompete(observation, fitness, phenotypes, obsMap, d, p);
    obsMap = updateMap(replaced,replacement,obsMap,fitness,observation,...
                        values,features, p.extraMapValues);
    
    % Illuminate with QD (but no visualization)
    acqCfg = p;acqCfg.display.illu = false;
    acqMap = illuminate(obsMap,acqFunction,acqCfg,d);

    %% 3 - Select Infill Samples
    % The next samples to be tested are chosen from the acquisition map: a
    % sobol sequence is used to to evenly sample the map in the feature
    % dimensions. When evaluated solutions don't converge the next bin in
    % the sobol set is chosen.
    disp(['PE: ' int2str(nSamples) '| Evaluating New Samples']);
    % At first iteration initialize sobol sequence for sample selection
    if nSamples == p.numInitSamples
        if isfield(d,'commonSobolGen')
            sobSet = d.commonSobolGen;
        else
            sobSet  = scramble(sobolset(d.nDims,'Skip',1e3),'MatousekAffineOwen');
        end
        if isfield(d,'commonSobolGenPtr')
            sobPoint = commonSobolGenPtr;
        else
            sobPoint= 1;
        end
    end
    
    newValue = nan(p.infill.nAdditionalSamples, size(fitness,2)); % new values will be stored here
    noValue = any(isnan(newValue),2);
    
    while any(any(noValue))
        nNans = sum(noValue);
        nextGenes = nan(nNans,d.dof); % Create one 'blank' genome for each NAN
        
        % Identify (grab indx of NANs)
        nanIndx = 1:p.infill.nAdditionalSamples;  nanIndx = nanIndx(noValue);
        
        % Replace with next in Sobol Sequence
        newSampleRange = sobPoint:(sobPoint+nNans-1);
        mapLinIndx = sobol2indx(sobSet, newSampleRange, d, p, acqMap.edges);
        
        % Replace unreachable bins
        emptyCells = isnan(acqMap.fitness(mapLinIndx));
        while any(emptyCells)
            nEmptyCells = sum(emptyCells);
            mapLinIndx(emptyCells) = sobol2indx(sobSet,sobPoint:sobPoint+nEmptyCells-1, d, p, acqMap.edges);
            emptyCells = isnan(acqMap.fitness(mapLinIndx));
            sobPoint = sobPoint + nEmptyCells;
        end
        
        % Pull out chosen genomes from map
        [chosenI,chosenJ] = ind2sub(p.infill.featureResolution, mapLinIndx);
        for iGenes=1:nNans
            nextGenes(iGenes,:) = acqMap.genes(chosenI(iGenes),chosenJ(iGenes),:);
        end
        
        % Precise evaluation
        if ~isempty(nextGenes)
            measuredValue = fitnessFunction(nextGenes, d.fitfun);
            % Assign found values
            newValue(nanIndx,:) = measuredValue;
        end
        
        %if p.retryInvalid
            % Check for invalid or duplicate shapes
            %nanValue = any(isnan(newValue),2);
            %oldDuplicate = logical(false(1,size(nanValue,1)));
            %oldDuplicate(nanIndx) = any(pdist2(observation,nextGenes)==0);
            %newDuplicate = logical(false(1,size(nanValue,1)));
            %sampleDistances = pdist2(nextGenes,nextGenes);
            %sampleDistances = sampleDistances + diag(ones(1,size(sampleDistances,1)));
            %newDuplicate(nanIndx) = any(sampleDistances==0);
            %noValue = nanValue | oldDuplicate' | newDuplicate';
        %else
            % Do not try invalid shapes
            newValue(isnan(newValue(:,1)),:) = repmat([0 0],sum(isnan(newValue(:,1))),1);
            % We still have to skip samples from empty bins
            noValue = any(isnan(nextGenes),2);
        %end
        nextObservation(nanIndx,:) = nextGenes;         %#ok<AGROW>
        sobPoint = sobPoint + length(newSampleRange);   % Increment sobol sequence for next samples
    end
    
    % Add evaluated solutions to data set
    fitness = cat(1,fitness,newValue);
    observation = cat(1,observation,nextObservation);
    nSamples  = size(observation,1);
    
    % Assign new samples to parent pool as well
    %observation = cat(1,observation,nextObservation);
    
end % end acquisition loop


% Create prediction map
disp(['PE ' int2str(nSamples) ' | Training Prediction Models']); 
p.infill.modelParams.functionEvals = 100;
modelPred = trainGP(observation,fitness,p.infill.modelParams);

[predMap] = createPredictionMap(modelPred,fitnessFunction,p,d,'featureRes',p.predMapResolution);

end

