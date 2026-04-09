
%% Parameter Preparation
num_angles = 360; % 投影角度数
scan_angles = 2*pi; % 扫描角度
angles = linspace(0, scan_angles, num_angles);

DSD = 500; 
DSO = 200; 
DSD = expandToLength(DSD, num_angles);
DSO = expandToLength(DSO, num_angles);

geo = defaultGeometry('nVoxel', [128,128,128], 'sVoxel', [64,64,64], 'nDetector', [128,128], 'sDetector', [64,64], 'mode', 'cone', 'DSD', DSD, 'DSO', DSO);
geo = checkGeo(geo, angles);

%% Module and Projection Simulation
phantom_vol = sheppLogan3D(geo.nVoxel,'Modified Shepp-Logan');
projections = simulate_projections(phantom_vol, angles, geo);



%% Length Extension Function
function [output] = expandToLength(input, targetLength)
    if length(input) == 1
        output = repmat(input, 1, targetLength);
    elseif length(input) == targetLength
        output = input;
    else
        error('Length mismatch! Must be 1 or %d', targetLength);
    end
end