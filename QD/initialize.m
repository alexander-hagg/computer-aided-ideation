function [map,fitnessFunction] = initialize(constraints,d,p)
%INITIALIZE Initialize samples, fitness function, including constraints

fitnessFunction = @(x,fitfun) objective(x,fitfun,d,[],p.penaltyWeight,p.driftThreshold);
if ~isempty(constraints)
    disp(['Adding constraints to the fitness function']);
    fitnessFunction = @(x,fitfun) objective(x,d.fitfun,d,constraints,p.penaltyWeight,p.driftThreshold);
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
[fitness, values, phenotypes]       = fitnessFunction(initSamples,d.fitfun); 

map                                 = createMap(d, p);
[replaced, replacement, features]   = nicheCompete(initSamples, fitness, phenotypes, map, d, p);
map                                 = updateMap(replaced,replacement,map,fitness,initSamples,values,features,p.extraMapValues);
end

