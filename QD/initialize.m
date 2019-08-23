function [map,fitnessFunction] = initialize(constraints,d,p,surrogateAssistance)
%INITIALIZE Initialize samples, fitness function, including constraints

if surrogateAssistance
    fitnessFunction = @(x,fitfun,varCoef) objective(x,fitfun,d,[],p.penaltyWeight,p.driftThreshold,varCoef);
else
    fitnessFunction = @(x) objective(x,d.fitfun,d,[],p.penaltyWeight,p.driftThreshold,0);
end

if ~isempty(constraints)
    disp(['Adding constraints to the fitness function']);
    if surrogateAssistance
        fitnessFunction = @(x,fitfun,varCoef) objective(x,fitfun,d,constraints,p.penaltyWeight,p.driftThreshold,varCoef);
    else
        fitnessFunction = @(x) objective(x,d.fitfun,d,constraints,p.penaltyWeight,p.driftThreshold,0);
    end
    initSamples = [];
    for it1=1:length(constraints)
        initSamples = [initSamples; constraints{it1}.members];
    end
else
    disp(['Initializing space filling sample set']);
    sobSequence         = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');
    sobPoint            = 1;
    initSamples         = range(d.ranges).*sobSequence(sobPoint:(sobPoint+p.numInitSamples)-1,:)+d.ranges(1);
end
if surrogateAssistance
    [fitness, values, phenotypes]       = fitnessFunction(initSamples,d.fitfun,d.varCoef); 
else
    [fitness, values, phenotypes]       = fitnessFunction(initSamples); 
end

map                                 = createMap(d, p);
[replaced, replacement, features]   = nicheCompete(initSamples, fitness, phenotypes, map, d, p);
map                                 = updateMap(replaced,replacement,map,fitness,initSamples,values,features,p.extraMapValues);
end

