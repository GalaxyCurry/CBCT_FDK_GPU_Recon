下面是一个为 GitHub 项目编写的 `README.md` 示例。该项目旨在通过 GPU 加速实现 CBCT 三维重建，并帮助学习者掌握重建算法与 CUDA 编程。

```markdown
# CBCT-Reconstruction-Learn

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**CBCT-Reconstruction-Learn** 是一个用于锥束计算机断层扫描（CBCT）三维重建的学习型项目。它实现了多种经典与前沿的重建算法，并利用 **GPU（CUDA）** 进行加速，旨在帮助开发者深入理解重建原理、并行计算模式以及 CUDA 编程实践。

> 🎯 **目标**：在实现完整 CBCT 重建流程的同时，提供一个可扩展、注释丰富、性能可控的代码库，让学习算法与 GPU 加速变得同样有趣。

## 📌 特性

- ✅ **多种重建算法**：FDK（Feldkamp-Davis-Kress）、ART（代数重建技术）、SART（联合代数重建）、MLEM（最大似然期望最大化）等。
- ⚡ **GPU 加速**：核心算子（反投影、投影、滤波、迭代更新）使用 CUDA 编写，显著提升重建速度。
- 🧩 **模块化设计**：算法基类、投影几何、探测器模型、加速器抽象层便于扩展和教学。
- 📊 **可视化与评估**：内置重建切片查看器、均方误差（MSE）、峰值信噪比（PSNR）评估工具。
- 📖 **教育优先**：代码中附有详细注释，并包含算法推导文档和 CUDA 优化技巧说明。
- 🔧 **跨平台支持**：Linux / Windows（通过 CMake + MSVC）测试通过。

## 🧠 已实现算法

| 算法 | 类别 | GPU 加速 | 状态 |
|------|------|----------|------|
| FDK | 解析法（滤波反投影） | ✅ 反投影、斜坡滤波 | 稳定 |
| SART | 迭代法（联合代数） | ✅ 投影/反投影对 | 稳定 |
| ART | 迭代法（逐射线） | ✅ 并行更新 | 开发中 |
| MLEM | 统计迭代法 | ✅ 投影/反投影 | 规划 |

## 🏗️ 构建与安装

### 依赖项
- CMake 3.15+
- CUDA Toolkit 11.0+（支持 Compute Capability 5.0+）
- C++17 编译器（GCC 8+ / MSVC 2019+）
- 可选：Python 3.8+（用于测试脚本与可视化）

### 编译步骤

```bash
git clone https://github.com/yourusername/CBCT-Reconstruction-Learn.git
cd CBCT-Reconstruction-Learn
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

若需生成 Python 绑定（pybind11）：
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=ON
```

## 🚀 快速开始

### 使用命令行工具

```bash
# 运行 FDK 重建（默认使用 GPU）
./bin/cbct_recon --input data/projection.raw --output recon.raw --algorithm fdk --size 256 256 256

# 使用 SART 迭代重建，设置 20 次迭代
./bin/cbct_recon --input proj/ --output sart_result.raw --algorithm sart --iterations 20
```

### Python 示例

```python
import cbct_recon as cr

proj = cr.load_projections("data/proj_360.raw", shape=(360, 512, 512))
geo = cr.Geometry(source_distance=750, detector_distance=400, angles=360)
recon = cr.reconstruct(proj, geo, algorithm="fdk", gpu=True)
cr.save_volume(recon, "output.raw")
```

### 测试数据集

- 使用 [Shepp-Logan 体模](https://github.com/...) 生成的合成投影（见 `data/generate_synthetic.py`）
- 公开数据集：[Mayo Clinic CT Data](https://wiki.cancerimagingarchive.net/display/Public/Mayo+Clinic+CT)，需转换为锥束几何

## 📂 项目结构

```
CBCT-Reconstruction-Learn/
├── include/          # 公共头文件（算法接口、几何、内存管理）
├── src/              # 核心实现
│   ├── algorithms/   # FDK, SART, ART 等实现
│   ├── cuda/         # CUDA kernels 及加速器
│   ├── geometry/     # 投影矩阵、探测器模型
│   └── io/           # 投影与体数据读写
├── examples/         # 命令行工具与演示脚本
├── tests/            # 单元测试（Catch2 + CUDA 测试）
├── docs/             # 算法推导、CUDA 优化笔记
└── python/           # Python 绑定（pybind11）
```

## 🧪 验证与性能

- **正确性**：使用合成投影与解析解对比，FDK 与 SART 重建的峰值信噪比 > 40dB（理想条件）。
- **性能**（NVIDIA RTX 3060, 体积 256³，投影 360 张 512×512）：
  - FDK：< 1.2 秒（GPU） vs 45 秒（单核 CPU）
  - SART（20 次迭代）：28 秒（GPU） vs 不可行（CPU）

> 详细的基准测试请参考 [docs/performance.md](docs/performance.md)

## 📖 学习资源

本仓库不仅提供代码，还包含大量学习材料：

- **[algorithms/README.md](src/algorithms/README.md)**：每个算法的数学原理与伪代码。
- **[cuda/README.md](src/cuda/README.md)**：CUDA 优化技巧（共享内存、纹理内存、流并发）。
- **[docs/iterative_acceleration.md](docs/iterative_acceleration.md)**：迭代算法加速方法（有序子集、块迭代）。
- **代码注释**：所有 CUDA kernel 都包含边界条件处理和并行模式分析。

## 🤝 贡献指南

欢迎贡献新算法、优化现有 kernel 或完善文档！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/awesome-alg`)
3. 确保 CUDA 代码通过 `cuda-memcheck` 和单元测试
4. 提交 Pull Request

请遵守 [代码风格](docs/code_style.md)（基于 clang-format）和 [提交消息规范](docs/commit_convention.md)。

## 📄 许可证

[MIT License](LICENSE) – 鼓励学习和商用，但请保留版权声明。

## 🙏 致谢

- 受 [The ASTRA Toolbox](https://www.astra-toolbox.com/) 和 [TIGRE](https://github.com/CERN/TIGRE) 启发
- 感谢 NVIDIA CUDA 教学团队提供的示例代码

## 📬 联系方式

- 项目维护者：Your Name (your.email@example.com)
- 讨论与问题：请使用 [GitHub Issues](https://github.com/yourusername/CBCT-Reconstruction-Learn/issues)

---

**Happy Learning & Coding!** 🚀
```

> 💡 **提示**：实际使用时请替换 `yourusername`、邮箱地址以及可能的外部链接。若项目未实现 Python 绑定或某些算法，请相应修改“已实现算法”表格和构建选项。此 README 强调“学习”目的，适合作为课程项目或个人学习仓库的文档模板。
