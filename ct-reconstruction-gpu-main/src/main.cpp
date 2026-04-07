#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <stdexcept>
#include <limits>
#include <H5Cpp.h>
#include <CL/cl.h>

#ifdef _WIN32
#  include <windows.h>
#  include <tchar.h>
#  include <sstream>
std::string wstring_to_string(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}
#else
#  include <unistd.h>
#endif

#include "reconstruction.hpp"

using namespace H5;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;


std::string getCpuModelName() {
#ifdef _WIN32
    
    HKEY hKey;
    const wchar_t* regPath = L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
    
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, regPath, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        WCHAR cpuName[256] = { 0 };
        DWORD dataSize = sizeof(cpuName);

        RegQueryValueExW(hKey, L"ProcessorNameString", NULL, NULL, (LPBYTE)cpuName, &dataSize);
        RegCloseKey(hKey);

        return wstring_to_string(cpuName);
    }
    return "Unknown CPU (Windows)";

#else
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("model name") != std::string::npos) {
            return line.substr(line.find(':') + 2);
        }
    }
    return "Unknown CPU (Linux)";
#endif
}

std::string getComputerName() {
    char buf[1024] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf); GetComputerNameA(buf, &sz);
#else
    gethostname(buf, sizeof(buf));
#endif
    return std::string(buf);
}

std::string getGpuModelName() {
    cl_platform_id plat; cl_uint np = 0;
    if (clGetPlatformIDs(1,&plat,&np)!=CL_SUCCESS||np==0) return "No OpenCL platform";
    cl_device_id dev; cl_uint nd = 0;
    if (clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,&nd)!=CL_SUCCESS||nd==0) return "No GPU";
    size_t sz=0; clGetDeviceInfo(dev,CL_DEVICE_NAME,0,nullptr,&sz);
    std::string name(sz,'\0'); clGetDeviceInfo(dev,CL_DEVICE_NAME,sz,&name[0],nullptr);
    if (!name.empty()&&name.back()=='\0') name.pop_back();
    return name;
}

static double elapsed(Clock::time_point t0){ return Seconds(Clock::now()-t0).count(); }

static void writeVol(H5File& f,const std::string& name,
                     const std::vector<float>& data,hsize_t NX,hsize_t NY){
    hsize_t dims[3]={NX,NX,NY}; DataSpace sp(3,dims);
    DataSet ds=f.createDataSet(name,PredType::NATIVE_FLOAT,sp);
    ds.write(data.data(),PredType::NATIVE_FLOAT);
    std::cout<<"  Saved: "<<name<<"\n";
}

static void printValueRange(const std::string& label,const std::vector<float>& v){
    float mn=*std::min_element(v.begin(),v.end());
    float mx=*std::max_element(v.begin(),v.end());
    float mean=0.f; for(auto x:v) mean+=x; mean/=(float)v.size();
    std::cout<<"  "<<std::left<<std::setw(22)<<label
             <<"  min="<<std::setw(12)<<mn
             <<"  max="<<std::setw(12)<<mx
             <<"  mean="<<mean<<"\n";
}

static std::vector<float> tryLoadPythonReference(H5File& file){
    for(auto& name:{"Reconstruction","reconstruction","Reconstruction_Python","recon","volume"}){
        try{
            DataSet ds=file.openDataSet(name);
            DataSpace sp=ds.getSpace(); hsize_t dims[3];
            sp.getSimpleExtentDims(dims);
            std::vector<float> ref(dims[0]*dims[1]*dims[2]);
            ds.read(ref.data(),PredType::NATIVE_FLOAT);
            std::cout<<"  [Python reference loaded from: '"<<name<<"']\n";
            return ref;
        }catch(...){}
    }
    return {};
}

int main(int argc, char* argv[]){
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"   CT Reconstruction - GPU Lab Project\n"
             <<"   FDK Cone-Beam CT - 5-pipeline benchmark\n";

    std::string path="D:\\tasks\\CT_Recon\\CBCT_FDK_GPU_Recon\\ct-reconstruction-gpu-main\\dataset\\proj_shepplogan128.hdf5";
    bool gpuOnly = false;
    for(int i=1;i<argc;++i){
        std::string arg=argv[i];
        if(arg=="--gpu-only") gpuOnly=true;
        else path=arg;
    }
    if(gpuOnly)
        std::cout<<"[MODE] GPU-only: CPU reconstruction skipped.\n"
                 <<"       GPU-Buffer used as reference for MSE.\n\n";

    try{
        std::cout<<"[INFO] Loading:\n  "<<path<<"\n\n";
        H5File file(path.c_str(),H5F_ACC_RDONLY);
        int P,W,H,NX,NY; float voxelSize,pixelSize,SDD,SOD;
        file.openDataSet("num_projs")      .read(&P,        PredType::NATIVE_INT);
        file.openDataSet("detector_width") .read(&W,        PredType::NATIVE_INT);
        file.openDataSet("detector_height").read(&H,        PredType::NATIVE_INT);
        file.openDataSet("Volumen_num_xz") .read(&NX,       PredType::NATIVE_INT);
        file.openDataSet("Volumen_num_y")  .read(&NY,       PredType::NATIVE_INT);
        file.openDataSet("voxelSize")      .read(&voxelSize,PredType::NATIVE_FLOAT);
        file.openDataSet("pixelSize")      .read(&pixelSize,PredType::NATIVE_FLOAT);
        file.openDataSet("SDD")            .read(&SDD,      PredType::NATIVE_FLOAT);
        file.openDataSet("SOD")            .read(&SOD,      PredType::NATIVE_FLOAT);

        DataSet projDS=file.openDataSet("Projection");
        DataSpace projSpace=projDS.getSpace(); hsize_t dims[3];
        projSpace.getSimpleExtentDims(dims);
        std::vector<float> projections(dims[0]*dims[1]*dims[2]);
        projDS.read(projections.data(),PredType::NATIVE_FLOAT);

        H5::Exception::dontPrint();
        auto volPython=tryLoadPythonReference(file);

        std::cout<<"[INFO] Dataset loaded\n"
                 <<"  Projections : "<<P<<"\n"
                 <<"  Detector    : "<<W<<" x "<<H<<" px\n"
                 <<"  Volume      : "<<NX<<" x "<<NX<<" x "<<NY<<" voxels\n"
                 <<"  SDD / SOD   : "<<SDD<<" / "<<SOD<<" mm\n"
                 <<"  voxelSize   : "<<voxelSize<<" mm\n"
                 <<"  pixelSize   : "<<pixelSize<<" mm\n\n";

        double projGB = (double)P*W*H*4*2/1e9;
        double volGB  = (double)NX*NX*NY*4/1e9;
        std::cout<<"[INFO] Estimated memory: projections="<<std::setprecision(2)
                 <<projGB<<" GB  volume="<<volGB<<" GB\n\n"
                 <<std::setprecision(6);

        double tCPU = 0.0;
        std::vector<float> volCPU;
        if(!gpuOnly){
            std::cout<<"[1/5] CPU Reconstruction (FDK reference)...\n";
            auto t0=Clock::now();
            volCPU=reconstructCPU(projections,P,W,H,NX,NY,voxelSize,pixelSize,SDD,SOD);
            tCPU=elapsed(t0);
            std::cout<<"  Done in "<<tCPU<<" s\n\n";
        } else {
            std::cout<<"[1/5] CPU Reconstruction — SKIPPED (--gpu-only mode)\n\n";
        }

        std::cout<<"[2/5] GPU-Buffer (filter CPU, backproject GPU global memory)...\n";
        auto t0=Clock::now();
        auto volBuf=reconstructGPU_Buffer(projections,P,W,H,NX,NY,voxelSize,pixelSize,SDD,SOD);
        double tBuf=elapsed(t0);
        std::cout<<"  Done in "<<tBuf<<" s\n\n";

        const std::vector<float>& ref = gpuOnly ? volBuf : volCPU;
        const std::string refName     = gpuOnly ? "GPU-Buffer" : "CPU";

        std::cout<<"[3/5] GPU-Image (filter CPU, backproject GPU image2d_t)...\n";
        t0=Clock::now();
        auto volImg=reconstructGPU_Image(projections,P,W,H,NX,NY,voxelSize,pixelSize,SDD,SOD);
        double tImg=elapsed(t0);
        std::cout<<"  Done in "<<tImg<<" s\n\n";

        std::cout<<"[4/5] GPU-Full (ramp filter + backproject both on GPU)...\n";
        t0=Clock::now();
        auto volFull=reconstructGPU_Buffer_Full(projections,P,W,H,NX,NY,voxelSize,pixelSize,SDD,SOD);
        double tFull=elapsed(t0);
        std::cout<<"  Done in "<<tFull<<" s\n\n";

        std::cout<<"[5/5] GPU-Local (local memory tiling + 3D NDRange)...\n";
        t0=Clock::now();
        auto volLocal=reconstructGPU_Local(projections,P,W,H,NX,NY,voxelSize,pixelSize,SDD,SOD);
        double tLocal=elapsed(t0);
        std::cout<<"  Done in "<<tLocal<<" s\n\n";

        std::cout<<" --------------------------------Validation--------------------------------\n\n";

        std::cout<<"  [Step 1] Python reference vs C++ CPU:\n";
        if(!volPython.empty()&&!volCPU.empty()){
            double msePyCPU=computeMSE(volPython,volCPU);
            std::cout<<"  MSE (Python vs C++ CPU) = "<<msePyCPU<<"\n";
            std::cout<<"  → "<<(msePyCPU<1e-2 ? "C++ CPU matches Python reference [OK]" : "WARNING: larger than expected.")<<"\n";
        } else if(gpuOnly){
            std::cout<<"  Skipped (--gpu-only mode, no CPU reconstruction)\n";
            std::cout<<"  → See 128-dataset run for Python vs CPU validation.\n";
        } else {
            std::cout<<"  Python reference not found in HDF5.\n";
            std::cout<<"  Run: python3 validate_and_plot.py\n";
        }
        std::cout<<"\n";




        double mseBuf  =gpuOnly?0.0:computeMSE(ref,volBuf);
        double mseImg  =computeMSE(ref,volImg);
        double mseFull =computeMSE(ref,volFull);
        double mseLocal=computeMSE(ref,volLocal);
        std::cout<<"  [Step 2] "<<refName<<" vs GPU pipelines:\n";
        if(!gpuOnly)
            std::cout<<"  MSE ("<<refName<<" vs GPU-Buffer) : "<<mseBuf  <<"  "<<(mseBuf  <1e-1 ? "[OK]" : "[FAIL]")<<"\n";
        std::cout<<"  MSE ("<<refName<<" vs GPU-Image)  : "<<mseImg  <<"  "<<(mseImg  <1e-1 ? "[OK]" : "[FAIL]")<<"\n";
        std::cout<<"  MSE ("<<refName<<" vs GPU-Full)   : "<<mseFull <<"  "<<(mseFull <1e-1 ? "[OK]" : "[FAIL]")<<"\n";
        std::cout<<"  MSE ("<<refName<<" vs GPU-Local)  : "<<mseLocal<<"  "<<(mseLocal<1e-1 ? "[OK]" : "[FAIL]")<<"\n";
        std::cout<<"  (< 1e-4: excellent  |  < 1e-2: good  |  < 1e-1: acceptable)\n\n";

        std::cout<<"  [Value Range] Min / Max / Mean:\n";
        if(!gpuOnly) printValueRange("CPU", volCPU);
        printValueRange("GPU-Buffer", volBuf);
        printValueRange("GPU-Image",  volImg);
        printValueRange("GPU-Full",   volFull);
        printValueRange("GPU-Local",  volLocal);
        std::cout<<"\n";






        std::cout<<"[INFO] Writing raw volumes for HDF5 packing...\n";
        {
            std::ofstream meta("../viz/volumes_meta.txt");
            meta << NX << " " << NY << "\n";
            meta << (gpuOnly ? "0" : "1") << "\n";
            meta.close();

            auto dumpBin = [](const std::string& name, const std::vector<float>& v){
                std::string fullPath = "D:/tasks/CT_Recon/CBCT_FDK_GPU_Recon/ct-reconstruction-gpu-main/viz/" + name;
                std::ofstream f(fullPath, std::ios::binary);
                f.write(reinterpret_cast<const char*>(v.data()),
                        v.size()*sizeof(float));
                std::cout<<"  Dumped: "<<name<<"\n";
            };

            if(!gpuOnly) dumpBin("vol_cpu.bin",   volCPU);
            dumpBin("vol_buf.bin",   volBuf);
            dumpBin("vol_img.bin",   volImg);
            dumpBin("vol_full.bin",  volFull);
            dumpBin("vol_local.bin", volLocal);
        }
        std::cout<<"  Run: python3 pack_hdf5.py\n\n";







        double bestTime=std::min<double>({tBuf,tImg,tFull,tLocal});
        auto tag=[&](double t)->const char*{return(t==bestTime)?"   fastest":"";};

        std::cout
            <<"--------------------- Benchmark Summary ---------------------\n"
            <<"  Computer : "<<getComputerName()<<"\n"
            <<"  CPU      : "<<getCpuModelName()<<"\n"
            <<"  GPU      : "<<getGpuModelName()<<"\n"
            <<"  -----------------------------------------------------------\n"
            <<"  Pipeline                  Time(s)         Speedup\n"
            <<"  -----------------------------------------------------------\n";
        if(!gpuOnly)
            std::cout<<"  [1] CPU (reference)        "<<std::setw(10)<<tCPU<<"      1.000x\n";
        else
            std::cout<<"  [1] CPU (reference)        "<<std::setw(10)<<"N/A"<<"      N/A (--gpu-only)\n";
        double refTime = gpuOnly ? tBuf : tCPU;
        std::cout<<"  [2] GPU Buffer             "<<std::setw(16)<<tBuf <<std::setw(10)<<refTime/tBuf <<"x"<<tag(tBuf) <<"\n"
                 <<"  [3] GPU Image              "<<std::setw(16)<<tImg <<std::setw(10)<<refTime/tImg <<"x"<<tag(tImg) <<"\n"
                 <<"  [4] GPU Full               "<<std::setw(16)<<tFull<<std::setw(10)<<refTime/tFull<<"x"<<tag(tFull)<<"\n"
                 <<"  [5] GPU Local(tiling)      "<<std::setw(16)<<tLocal<<std::setw(10)<<refTime/tLocal<<"x"<<tag(tLocal)<<"\n"
                 <<"----------------------------------------------------\n";
    }
    catch(const H5::Exception& e){std::cerr<<"HDF5 Error: "<<e.getDetailMsg()<<"\n";return -1;}
    catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<"\n";return -1;}
    return 0;
}
