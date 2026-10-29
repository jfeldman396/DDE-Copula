function B = sim_block_loadings(J, K1, blockSize, posMag, negMag, noiseSD, seed)
    %SIM_BLOCK_SIGNED_LOADINGS Simulate a J x K1 loading matrix with signed blocks.
    %
    % Every `blockSize` rows load strongly and positively on one unique column,
    % and strongly and negatively on a different column. Remaining entries are noise.
    %
    % Inputs:
    % J - number of rows
    % K1 - number of columns
    % blockSize - rows per block (use 10 for your request)
    % posMag - magnitude of positive strong loading (e.g., 2 or 3)
    % negMag - magnitude of negative strong loading (e.g., 2 or 3)
    % noiseSD - std dev of background noise (e.g., 0.05)
    % seed - RNG seed (optional; [] to skip)
    %
    % Output:
    % B - J x K1 matrix
    %
    % Example:
    % B = sim_block_signed_loadings(200, 30, 10, 3, 3, 0.05, 1);
    
    if nargin < 7, seed = []; end
    if nargin < 6 || isempty(noiseSD), noiseSD = 0; end
    if nargin < 5 || isempty(negMag), negMag = posMag; end
    if nargin < 4 || isempty(posMag), posMag = 3; end
    if nargin < 3 || isempty(blockSize), blockSize = 10; end
    
    if ~isempty(seed)
    rng(seed);
    end
    
    % Base noise
    B = noiseSD * randn(J, K1);
    
    nBlocks = ceil(J / blockSize);
    
    % Ensure we have enough columns for unique positive columns across blocks
    if K1 < nBlocks
    error('K1=%d must be >= number of blocks=%d for unique positive columns.', K1, nBlocks);
    end
    
    % Positive columns: unique per block (wrap not allowed due to check above)
    posCols = 1:nBlocks;
    
    % Negative columns: different from posCol for each block
    negCols = zeros(1, nBlocks);
    for b = 1:nBlocks
    candidates = setdiff(1:K1, posCols(b));
    negCols(b) = candidates(randi(numel(candidates)));
    end
    
    % Fill blocks
    for b = 1:nBlocks
    r1 = (b-1)*blockSize + 1;
    r2 = min(b*blockSize, J);
    rows = r1:r2;
    
    B(rows, posCols(b)) = B(rows, posCols(b)) + posMag;
    B(rows, negCols(b)) = B(rows, negCols(b)) - negMag;
end