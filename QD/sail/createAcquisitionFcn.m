function acqFunction = createAcquisitionFcn(fitnessFunction,model,d)
%CREATEACQUISITIONFCN Packages GP models into easily used acquisition function
%
% Syntax:  acqFunction 
%
% Inputs:
%    gpModel - cell - one or more gaussian process models
%    d              - Domain description struct
%    .express       - genotype->phenotype conversion function
%
% Outputs:
%    acqFunction - anonymous function that takes genome as input and
%

% Author: Adam Gaier, Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (BRSU)
% email: adam.gaier@h-brs.de, alexander.hagg@h-brs.de
% Jun 2017; Last revision: 23-Aug-2019

%------------- BEGIN CODE --------------

acqFunction = @(x,d) fitnessFunction( x, feval('predictGP', model, x));

%[adjustedFitness, values, phenotypes]
%------------- END OF CODE --------------
end

