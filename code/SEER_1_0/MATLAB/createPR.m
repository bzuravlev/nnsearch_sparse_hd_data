% Compute precision and recall vectors for a given similarrity matrix
% and binary ground truth matrices. 
%
% S ... similarity matrix
% GThard ... ground truth matching matrix: 1 at places that must be matched,
%            else 0; must have the same shape as S
% GTsoft ... ground truth places that CAN be matched without penalty; must
%            have the same shape as S
%
% P ... precision vector
% R ... recall vector
%
% peer.neubert@etit.tu-chemnitz.de, 2022
function [P, R] = createPR(S, GThard, GTsoft)
    
    % remove soft-but-not-hard-entries
    if ~isempty(GTsoft)
        S(GTsoft & ~GThard) = min(S(:));
    end
    
    GT = logical(GThard); % ensure logical-datatype
    
    % init precision and recall vectors
    R = 0;
    P = 1;
    
    % select start and end treshold
    startV = max(S(:)); % start-value for treshold
    endV = min(S(:)); % end-value for treshold
    
    % iterate over different thresholds
    for i=linspace(startV, endV, 100) 
        B = S>=i; % apply threshold
        
        TP = nnz( GT & B ); % true positives
        FN = nnz( GT & (~B) ); % false negatives
        FP = nnz( (~GT) & B ); % false positives
        
        P(end+1) = TP/(TP + FP); % precision
        R(end+1) = TP/(TP + FN); % recall
    end
end

