% Create a Gaussian Random Projection matrix PP of size(nInDIms, nOutDims).
%
% nInDims ... number of input dimensions (rows in PP)
% nOutDims ... number of output dimensions (columns in PP)
% seed ... is the seed for the random number generator
%
% If nInDIms==nOutDims, the output is orthogonalized (this takes some time
% for large dimensions).
%
% peer.neubert@etit.tu-chemnitz.de, 2022
function PP = createGRPMatrix(nInDims, nOutDims, seed)
    rng(seed);  
    if nInDims==nOutDims
        PP = randn(round(nInDims*1.2), nOutDims, 'single'); % some dims more ...
        PP = orth(PP')'; % ... to be able to select nInDims many
        assert(size(PP,1)==nInDims);
    else
       PP = randn(round(nInDims), nOutDims, 'single');
    end
    PP = normc(PP); 
end