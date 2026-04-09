
% ========== 几何参数（需要改成自己的值） ==========
num_projs      = 360;          % 投影数量 P
detector_width = 128;          % 探测器宽度 W
detector_height= 128;          % 探测器高度 H
Volumen_num_z  = 128;          % 重建体 Z 维度 NZ
Volumen_num_x  = 128;          % 重建体 X 维度 NX
Volumen_num_y  = 128;          % 重建体 Y 维度 NY
voxelSize      = 0.5;          % 体素大小 mm
pixelSize      = 0.5;          % 探测器像素大小 mm
SDD            = 500;          % 源-探测器距离 mm
SOD            = 200;          % 源-旋转中心距离 mm

% ========== 投影数据（3D 数组：W × H × P） ========== 

volume_reference = rand(Volumen_num_y, Volumen_num_xz, Volumen_num_xz, 'single');

% ========== 输出文件名 ==========
outputFilename = 'proj_shepplogan128.hdf5';




fprintf('正在生成 HDF5 文件: %s\n', outputFilename);

% --------------------
% 写入 整型 参数
% --------------------
h5create(outputFilename, '/num_projs',       1, 'Datatype', 'int32');
h5write( outputFilename, '/num_projs',       int32(num_projs));

h5create(outputFilename, '/detector_width',  1, 'Datatype', 'int32');
h5write( outputFilename, '/detector_width',  int32(detector_width));

h5create(outputFilename, '/detector_height', 1, 'Datatype', 'int32');
h5write( outputFilename, '/detector_height', int32(detector_height));

h5create(outputFilename, '/Volumen_num_xz',  1, 'Datatype', 'int32');
h5write( outputFilename, '/Volumen_num_xz',  int32(Volumen_num_xz));

h5create(outputFilename, '/Volumen_num_y',   1, 'Datatype', 'int32');
h5write( outputFilename, '/Volumen_num_y',   int32(Volumen_num_y));

% --------------------
% 写入 浮点型 参数（C++ 需要 float / single）
% --------------------
h5create(outputFilename, '/voxelSize', 1, 'Datatype', 'single');
h5write( outputFilename, '/voxelSize', single(voxelSize));

h5create(outputFilename, '/pixelSize', 1, 'Datatype', 'single');
h5write( outputFilename, '/pixelSize', single(pixelSize));

h5create(outputFilename, '/SDD',       1, 'Datatype', 'single');
h5write( outputFilename, '/SDD',       single(SDD));

h5create(outputFilename, '/SOD',       1, 'Datatype', 'single');
h5write( outputFilename, '/SOD',       single(SOD));

% --------------------
% 写入投影数据：[W, H, P]
% 关键：single(float32)
% --------------------
projDims = size(projections);
h5create(outputFilename, '/Projection', projDims, 'Datatype', 'single');
h5write( outputFilename, '/Projection', single(projections));


fprintf('✅ 生成完成！可直接用于 ct-recon 项目\n');
fprintf('文件路径：%s\n', outputFilename);