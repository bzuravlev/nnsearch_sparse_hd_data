% Demonstration of SEER on an example dataset. The actual SEER
% implementation is in runSEER.m
%
% peer.neubert@etit.tu-chemnitz.de, 2022
function demo_SEER

    fprintf('Running on Gardens Point Walking day_left vs. night_right\n');
    database_path = 'data/HDC_DELF_GardensPointWalking_day_left.mat';
    query_path = 'data/HDC_DELF_GardensPointWalking_night_right.mat';
    gtPath = 'data/groundTruth_GPW_DL_NR.mat';
    nInDims = 4096; % dimensionality of the input descriptors
    
    % prepare projection
    fprintf('Prepare projection matrix (needs only be computed once)\n');
    PP = createGRPMatrix(nInDims, 4096, 0);
        
    % parameters
    k = 50; 
    lambda = 2;
    d_M = 200;
    
    %% process database 
    fprintf('Load database\n');               
    DB = load(database_path, 'Y');     
    DB.Y = DB.Y*PP;     % project     
    DB_mean = mean(DB.Y);    
    DB_Y = double(DB.Y-DB_mean);  % standardize
                        
    % run SEER
    M = []; 
    rng(873734);    
    fprintf('SEER: first run on database\n');    
    [M, DB_SEER_run1] = runSEER(M, DB_Y, 1, d_M, k, lambda); % run SEER with adding new exemplars

    % second run to generate output descriptor    
    fprintf('SEER: second run on database\n');    
    [~, DB_SEER_run2] = runSEER(M, DB_Y, 0, d_M, k, lambda); % run SEER without adding new exemplars
    

    %% process query
    fprintf('Load query\n');  
    Q = load(query_path, 'Y');                            
    Q.Y = Q.Y*PP; % project 
    Q_Y = double(Q.Y - DB_mean); % standardize with DB mean
    
    % run SEER, generates output descriptor in a single run
    fprintf('SEER: run on query\n');   
    [~,Q_SEER] = runSEER(M, Q_Y, 0, d_M, k, lambda);% run SEER without adding new exemplars


    %% evaluate
    S = normr(DB_SEER_run2)*normr(Q_SEER)'; % this is a sparse matrix
    
    load(gtPath, 'GT');
    [P,R] = createPR(S,GT.GThard, GT.GTsoft); 
    AUC = trapz(R,P);
                
    fprintf('The area under the precision-recall curve is: %0.4f\n', AUC);
             
    
end

