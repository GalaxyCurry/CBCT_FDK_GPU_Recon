function projections = simulate_projections(vol, angles, geo)

    DSD = geo.DSD(1,1);
    DSO = geo.DSO(1,1);
    vol_size = geo.nVoxel;
    voxel_size = geo.dVoxel;

    [n, m, p] = size(vol); 
    num_angles = length(angles);
    projections = zeros(m, p, num_angles, 'single');
    
    % ========== 体模物理范围计算 ==========
    phys_size = vol_size .* voxel_size; % 体模物理尺寸 [mm]
    phys_min = -phys_size / 2; % 体模物理坐标最小值（中心在原点）
    phys_max = phys_size / 2;  % 体模物理坐标最大值
    % 体模轴对齐包围盒
    bbox_min = [phys_min(1), phys_min(2), phys_min(3)];
    bbox_max = [phys_max(1), phys_max(2), phys_max(3)];
    
    % ========== 探测器参数计算 ==========
    % 探测器像素尺寸（基于几何放大率）
    mag_factor = DSD / DSO; 
    detector_pixel_size = voxel_size(1) * mag_factor; 
    % 探测器物理范围
    detector_half_width = (p/2) * detector_pixel_size; % 列方向半宽
    detector_half_height = (m/2) * detector_pixel_size; % 行方向半高
    
    % ========== 探测器网格生成（行列顺序+步长） ==========
    u_vals = linspace(-detector_half_width, detector_half_width, p); % 列方向（u）：p个点
    v_vals = linspace(-detector_half_height, detector_half_height, m); % 行方向（v）：m个点
    [U, V] = meshgrid(u_vals, v_vals); % U: 列坐标，V: 行坐标（匹配探测器尺寸）
    
    % 逐角度生成投影
    for ang_idx = 1:num_angles
        beta = angles(ang_idx);
        % 旋转矩阵（绕z轴旋转）
        R = [cos(beta)  sin(beta) 0; 
             -sin(beta) cos(beta) 0;
             0          0         1];
        
        proj = zeros(m, p); % 当前角度的投影
        
        % 逐探测器像素计算投影值
        for det_row = 1:m 
            for det_col = 1:p 
                % 1. 源点和探测器点物理坐标
                source_phys = [0; -DSO(1,1); 0]; % 源位置（旋转中心左侧DDR处）
                detector_phys = [U(det_row, det_col); DSD - DSO; V(det_row, det_col)]; % 探测器平面位置
                
                % 2. 旋转源点和探测器点（模拟投影角度）
                source_rot = R * source_phys;
                detector_rot = R * detector_phys;
                
                % 3. 射线方向向量（归一化）
                ray_dir = detector_rot - source_rot;
                ray_dir_norm = ray_dir / norm(ray_dir);
                
                % ========== 射线与体模包围盒的交点计算 ==========
                % 轴对齐包围盒交点计算（正确的t_min/t_max）
                t_near = -inf;
                t_far = inf;
                % 遍历x/y/z轴计算交点
                for axis = 1:3
                    if ray_dir_norm(axis) == 0
                        % 射线平行于当前轴，检查是否在包围盒内
                        if source_rot(axis) < bbox_min(axis) || source_rot(axis) > bbox_max(axis)
                            t_near = inf; % 射线在包围盒外，无交点
                            break;
                        end
                    else
                        % 计算当前轴的交点t值
                        t1 = (bbox_min(axis) - source_rot(axis)) / ray_dir_norm(axis);
                        t2 = (bbox_max(axis) - source_rot(axis)) / ray_dir_norm(axis);
                        % 确保t1 < t2
                        if t1 > t2
                            temp = t1; t1 = t2; t2 = temp;
                        end
                        % 更新全局t范围
                        t_near = max(t_near, t1);
                        t_far = min(t_far, t2);
                    end
                end
                
                % 检查交点有效性
                if t_near > t_far || t_far < 0
                    proj(det_row, det_col) = 0;
                    continue;
                end
                
                % 确保步进范围从t_near开始（避免负数）
                t_start = max(t_near, 0);
                t_end = t_far;
                
                % 4. 射线步进采样（Siddon简化版）
                num_steps = 200;
                step_size = (t_end - t_start) / num_steps;
                ray_sum = 0;
                
                for t = t_start:step_size:t_end
                    % 射线当前位置（物理坐标）
                    point_phys = source_rot + t * ray_dir_norm;
                    
                    % 转换为体素索引（物理坐标 → 体素索引）
                    idx_x = round((point_phys(1) - phys_min(1)) / voxel_size(1)) + 1;
                    idx_y = round((point_phys(2) - phys_min(2)) / voxel_size(2)) + 1;
                    idx_z = round((point_phys(3) - phys_min(3)) / voxel_size(3)) + 1;
                    
                    % 检查索引有效性
                    if idx_x >= 1 && idx_x <= n && ...
                       idx_y >= 1 && idx_y <= m && ...
                       idx_z >= 1 && idx_z <= p
                        ray_sum = ray_sum + vol(idx_x, idx_y, idx_z) * step_size;
                    end
                end
                
                proj(det_row, det_col) = ray_sum;
            end
        end
        
        projections(:,:,ang_idx) = proj;
    end
end
