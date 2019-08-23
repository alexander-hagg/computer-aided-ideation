function [predMap, percImproved, percImprovement] = createPredictionMap(gpModel,fitnessFunction,p,d, varargin)
%createPredictionMap - Produce prediction map from surrogate model
%
% Syntax:  predictionMap = createPredictionMap(gpModels,p,d)
%
% Inputs:
%    gpModels   - GP model produced by SAIL
%    p          - SAIL hyperparameter struct
%    d          - Domain definition struct
%
% Outputs:
%    predMap - prediction map
%    .fitness     [Rows X Columns]
%    .genes       [Rows X Columns X GenomeLength]
%    .'otherVals' [Rows X Columns]
%    percImproved    - percentage of children which improved on elites
%
% Example: 
%    p = sail;
%    d = af_Domain;
%    output = sail(d,p);
%    predMap = createPredictionMap(output.model,p,d,'featureRes',[50 50]);
%    viewMap(predMap.fitness,d, predMap.edges)
%
% Other m-files required: mapElites  nicheCompete updateMap d.createAcqFunction
%
% See also: sail, mapElites, runSail

% Author: Adam Gaier
% Bonn-Rhein-Sieg University of Applied Sciences (BRSU)
% email: adam.gaier@h-brs.de
% Jun 2017; Last revision: 03-Aug-2017
%
%------------- BEGIN CODE --------------
if nargin > 4; figHandleMap = varargin{1};end
if nargin > 5; figHandleTotalFit = varargin{2};end
if nargin > 6; figHandleMeanDrift = varargin{3};end

d.varCoef = 0; % no award for uncertainty
% Construct functions
acqFunction = createAcquisitionFcn(fitnessFunction,gpModel,d);

% Seed map with precisely evaluated solutions
observation = gpModel.trainInput;

[fitness,values,phenotypes] = acqFunction(observation);

predMap = createMap(d,p);
[replaced, replacement, features] = nicheCompete(observation, fitness, phenotypes, predMap, d, p);
predMap = updateMap(replaced,replacement,predMap,fitness,observation,...
                        values,features, p.extraMapValues);

% Illuminate based on surrogate models
predMap = illuminate(predMap,acqFunction,p,d,figHandleMap,figHandleTotalFit,figHandleMeanDrift);
    
%------------- END OF CODE --------------