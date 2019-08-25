function demoAE(app)
%% Get phenotypes from map
genes = app.map{1}.genes;
fitness = app.map{1}.fitness;
genes = reshape(genes,[],size(genes,3));
fitness = reshape(fitness,size(fitness,1)*size(fitness,2),[]);
fitness(any(isnan(genes')),:) = [];
genes(any(isnan(genes')),:) = [];

lowFitness = fitness<0.8;
fitness(lowFitness) = [];
genes(lowFitness,:) = [];

phenotypes = getPhenotype(genes,app.d{1});
%% Convert phenotypes into images

res = 50;
images = [];
for i=1:length(phenotypes)
    if ~mod(i,50); disp(i);end
    x = phenotypes{i}.Vertices(:,1)';
    y = phenotypes{i}.Vertices(:,2)';
    x = x(~isnan(x));
    y = y(~isnan(y));
    x(end+1) = x(1);
    y(end+1) = y(1);
    x = (x + 1)/2;
    y = (y+1)/2;
    centerX = (max(x)+min(x))/2;
    centerY = (max(y)+min(y))/2;
    x = x + (0.5-centerX);
    y = y + (0.5-centerY);
    x = x*res; y = y*res;
    bw = poly2mask(x,y,res,res);
    
    %im = imshow(bw);
    images(i,:,:) = single(bw);
    %drawnow;
end


%% Prepare data
rdata = permute(images,[2 3 1]);
imwrite(1-tile_image(rdata), 'report/res/samples.png');
rdata = reshape(rdata,size(rdata,1)*size(rdata,2),size(rdata,3));
rawdata = [];
rawdata(1,1,:,:) = rdata;
%size(rawdata)

[trainInd,valInd,testInd] = dividerand(size(rawdata,4),0.9,0.1,0);
train = rawdata(:,:,:,trainInd);
valid = rawdata(:,:,:,valInd);
data = single(cat(4, train, valid));

imdb.images.data = data;
imdb.images.set = vertcat(ones(size(train, 4), 1), ...
                          2*ones(size(valid, 4), 1));
                      
%% Train Variational Autoencoder

opts.optim = 'ADAM';
opts.gpus = [1];
opts.hiddenSizes = [res*res, 500, 2];
sfx = sprintfc('%d', opts.hiddenSizes);
sfx = [sfx{1} '-' sfx{2} '-' sfx{3}];
opts.expDir = fullfile('models', ['QD-' sfx '-' opts.optim]);

% optimization parameters
opts.train = struct() ;
opts.train.gpus = opts.gpus; 
opts.train.numEpochs = 200;
opts.train.batchSize = 1024;
opts.train.derOutputs = {'NLL', 1, 'KLD', 1};


% initialize model 
rng(0);
net = init_model(opts.hiddenSizes(1), opts.hiddenSizes(2), opts.hiddenSizes(3));

% start training
switch opts.optim
  case 'SGD'
    trainfn = @sgd_train;
    opts.train.learningRate = 0.0005;
  case 'ADAGRAD'
    trainfn = @adagrad_train;
    opts.train.learningRate = 0.01;
  case 'RMSPROP'
    trainfn = @rmsprop_train;
    opts.train.learningRate = 0.001;
  case 'ADAM'
    trainfn = @adam_train;
    opts.train.learningRate = 0.001;
end

startup;
[netOriginal, info] = trainfn(net, imdb, getBatch(opts), 'expDir', opts.expDir, ...
                      opts.train) ;

%% Show Manifold
net = netOriginal.copy;
net.conserveMemory = false; 
net.mode = 'test';

net.removeLayer('h1');
net.removeLayer('tanh1');
net.removeLayer('h2');
net.removeLayer('split');
net.removeLayer('sample');


ny = 25;
nx = 25;
%Ys = icdf('normal', linspace(0,1,ny+2), 0, 1); Ys = Ys(2:end-1);
%s = icdf('normal', linspace(0,1,nx+2), 0, 1); Xs = Xs(2:end-1);
Ys = linspace(-3,3,ny); 
Xs = linspace(-3,3,nx); 
[yy,xx] = meshgrid(Ys, Xs);
z = cat(2,yy(:),xx(:))';
z = single(reshape(z, [1, 1, size(z)]));

net.eval({'z', z});

prob = net.vars(net.getVarIndex('prob')).value;
prob = gather(squeeze(prob));

prob = reshape(prob, [res,res,size(prob,2)]);
prob = permute(prob, [2,1,3]);

%imagesc(tile_image(prob));
%axis image;
%caxis([0, 1]);
imwrite(1-tile_image(prob), 'report/res/manifold.png');

%% --------------------------------------------------------------------
function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;


%% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images} ;

