% Implementation of Sparse Exemplar Ensemble Representations (SEER) 
%
%    Peer Neubert and Stefan Schubert (2022), "SEER: Unsupervised and 
%    sample-efficient environment specialization of image descriptors", 
%    Proceeding of Robotics: Science And Systems, New York, USA
%
% This is a batch version that computes output descriptors for a batch of
% input descriptors Y.
% 
% M ... sparse exemplar memory, each COLUMN is an exemplar, can be empty. It
%       is important to allocate the appropriate amount of memory. If an
%       empty matrix is provided, then the memory is allocated within this
%       function.
% Y ... input descriptor, each row is a descriptor. Should be a distributed
%       representation.
% updatedM_flag ... toggels whether new rows/exemplars are added to M
% d_M ... number of non-zero elemente in the exemplars in M, default: 200
% k ... minimum number of exemplars per database descriptor, default: 50
% lambda ... factor on k for non-zero elements  in the output descriptor,
%            default: 2
% 
% M ... see input, will be updated if updatedM_flag is set
% DS ... sparse output descriptor
%
% peer.neubert@etit.tu-chemnitz.de, 2022
function [M, DS] = runSEER(M, Y, updateM_flag, d_M, k, lambda)

    % default parameters
    if ~exist('d_M','var') || isempty(d_M), d_M = 200; end
    if ~exist('k', 'var') || isempty(k), k = 50; end
    if ~exist('lambda', 'var') || isempty(lambda), lambda = 2; end    
    
    
    %% allocate memory
    % allocate memory for output descriptor
    DS = sparse([],[],[],size(Y,1), size(M,2), size(Y,1)*k);
    
    % Allocate memory for M if it is empty AND there has not already been 
    % allocated some memory 
    if isempty(M)
        propM = whos('M');
        if propM.bytes<=24 % 24 bytes is the minimum size of an empty sparse double matrix 
            M = sparse([],[],[],size(Y,2), 0, size(Y,1)*k*d_M); 
        end
    end
        
    %% L2 normalize each input descritptor
    Y = normr(Y);
    
    %% process each descriptor
    for i=1:size(Y,1)
        if mod(i, 100)==0; fprintf('  running image %d of %d (%d exemplars) \n', i, size(Y,1), size(M,2)); end
        
        t = tic;
        
        % compute similarity
        if ~isempty(M)            
            S = Y(i,:)*M; % dot product similarity                   
        else
            S = [];
        end
        
        % sparsify
        [knnSim, knnIdx] = maxk(S, lambda*k);


        if updateM_flag

            % apply threshold
            expectedSimilarity =  d_M / size(Y,2);    % activity=activity* size(Y,2)/nDimSamples;
            threshIdx = knnSim >= expectedSimilarity;
            knnIdxAboveThresh = knnIdx(threshIdx);
        
            % potentially create new patterns in M        
            if numel(knnIdxAboveThresh)< k
            
                nNewPatterns = k - numel(knnIdxAboveThresh); % the number of new exemplars           
                [PI, PV] = createExemplars(Y(i,:),d_M, nNewPatterns);
                newPatternIdx = size(M,2) + [1:size(PI,1)]; % indices of the new exemplars in M
                newPatternActivity = sum(PV.*PV, 2)'; % activity of the new exemplars
    
                for pIdx = 1:size(PI,1)
                    M(PI(pIdx,:), end+1) = PV(pIdx,:);
                end              
            else
                newPatternIdx = [];
                newPatternActivity = [];
                nNewPatterns = 0;
            end
        else
            newPatternIdx = [];
            newPatternActivity = [];
            nNewPatterns = 0;
        end

        % output descriptor is kNN results above thresh and new patterns
        nRemainingKNN = min(numel(knnIdx), (k*lambda)-nNewPatterns);
        descrIdx = [knnIdx(1:nRemainingKNN) newPatternIdx];
        descrActivity = [knnSim(1:nRemainingKNN) newPatternActivity];
                
        % create output descriptor
        DS(i, descrIdx) = descrActivity; 
       
    end
    

end

% Sample new exemplars
%
% inputY ... continuous input descriptor
%
% PI ... index matrix, each row is vector of size nDimSamples
% PV ... the values from inputY that correspond to these indexes 
%
function [PI, PV] = createExemplars(inputY,nDimSamples, nPatterns)

    PI = zeros(nPatterns, nDimSamples);
    for i=1:nPatterns
        
        if nDimSamples == size(inputY,2)
            PI(i,:) = 1:size(inputY,2); % use all dimensions
        else
            PI(i,:) = datasample(1:size(inputY,2), nDimSamples, 'Weights',abs(inputY),'Replace', false);
        end
        
    end
    PV = inputY(PI);

end