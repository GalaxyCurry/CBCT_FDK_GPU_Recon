function geo=defaultGeometry(varargin)
%%geo = defaultGeometry() generates a default geometry for tests.
% Optional parameters
% 'mode'      : 'parallel' or 'cone'
% 'nVoxel'    : 3x1 matrix of size of the image
% 'sVoxel'    : 3x1 matrix of size of the image
% 'nDetector' : 2x1 matrix of size of the detector
% 'sDetector' : 2x1 matrix of size of the detector
% 'DSD' : 1x1 matrix or 1×angleDim matrix
% 'DSO' : 1x1 matrix or 1×angleDim matrix

[mode,nVoxel,sVoxel,nDetector,sDetector,DSD,DSO]=parse_inputs(varargin{:});

if strcmp(mode,'cone')
    % VARIABLE                                   DESCRIPTION                    UNITS
    %-------------------------------------------------------------------------------------
    % Distances
    geo.DSD = DSD;                             % Distance Source Detector      (mm)
    geo.DSO = DSO;                              % Distance Source Origin        (mm)

    % Detector parameters
    geo.nDetector=nDetector;					% number of pixels              (px)
    geo.sDetector=sDetector;                    % total size of the detector    (mm)
    geo.dDetector=geo.sDetector./geo.nDetector; % size of each pixel            (mm)

    % Image parameters
    geo.nVoxel=nVoxel;                          % number of voxels              (vx)
    geo.sVoxel=sVoxel;                          % total size of the image       (mm)
    geo.dVoxel=geo.sVoxel./geo.nVoxel;          % size of each voxel            (mm)
    
    % Offsets
    geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)
    geo.offDetector=[0;0];                      % Offset of Detector            (mm)
    
    % Auxiliary
    geo.accuracy=1;
    geo.mode=mode;
end



%%
function [mode,nVoxel,sVoxel,nDetector,sDetector,DSD,DSO]=parse_inputs(varargin)
% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'mode','cone',@(x)(ischar(x)&&(strcmp(x,'parallel')||strcmp(x,'cone'))));
addParameter(p,'nVoxel',   [256;256;256],@(x)(isnumeric(x)&&all(floor(x)==x)&&max(size(x))==3));
addParameter(p,'sVoxel',   [64;64;64],@(x)(isnumeric(x)&&all(floor(x)==x)&&max(size(x))==3));
addParameter(p,'nDetector',[256;256],@(x)(isnumeric(x)&&all(floor(x)==x)&&max(size(x))==2));
addParameter(p,'sDetector',[64;64],@(x)(isnumeric(x)&&all(floor(x)==x)&&max(size(x))==2));
addParameter(p,'DSD',1);
addParameter(p,'DSO',1);

%execute
parse(p,varargin{:});
%extract
mode=p.Results.mode;
nVoxel=p.Results.nVoxel;
sVoxel=p.Results.sVoxel;
nDetector=p.Results.nDetector;
sDetector=p.Results.sDetector;
DSD=p.Results.DSD;
DSO=p.Results.DSO;

if size(nDetector,1)==1
    nDetector=nDetector.';
end
if size(sDetector,1)==1
    sDetector=sDetector.';
end
if size(nVoxel,1)==1
    nVoxel=nVoxel.';
end
if size(sVoxel,1)==1
    sVoxel=sVoxel.';
end
if strcmp(mode,'parallel')&&(all(nVoxel(2:3)~= nDetector))
    warning('In Parallel mode nVoxel(2:3) is generally equal to nDetector. Consider setting them equal for better reconstruction quality.');
end

end
end