function infill = infillParamSet(varargin)
%SAILPARAMSET infill configuration for surrogate-assistance

infill.nAdditionalSamples = 10;
infill.nAddSamplesPerIteration = 50;
infill.trainingMod        = 3;
infill.featureResolution  = [20 20];
infill.retryInvalid       = true;

end

