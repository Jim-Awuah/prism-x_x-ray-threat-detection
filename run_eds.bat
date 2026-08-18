@echo off
REM =========================================================================
REM  PRISM-X EDS Training Pipeline — GPU Machine (Windows CMD)
REM  RTX 5060 Laptop GPU / CUDA
REM
REM  HOW TO USE:
REM  1. Save this file as run_eds.bat in your repo root:
REM     C:\Users\gpuworker\prism-x\prism-x_x-ray-threat-detection\
REM  2. Open CMD and run:
REM     cd C:\Users\gpuworker\prism-x\prism-x_x-ray-threat-detection
REM     run_eds.bat
REM  3. The script runs all stages in order and logs each step.
REM     If any step fails it will stop and tell you which step failed.
REM =========================================================================

SET PYTHON=C:\Users\pryyy\miniforge3\python.exe
SET DATA=C:\Users\gpuworker\datasets\EDS_Dataset
SET REPO=C:\Users\gpuworker\prism-x\prism-x_x-ray-threat-detection

cd /d %REPO%

echo.
echo =========================================================================
echo  STEP 0 — Verifying EDS dataset
echo =========================================================================
%PYTHON% -c "from pathlib import Path; root=Path(r'%DATA%'); [print(d, len(list((root/d/'image').glob('*.jpg'))), 'images,', len(list((root/d/'txt').glob('*.txt'))), 'annotations') for d in ['domain1','domain2','domain3']]"
if %errorlevel% neq 0 (
    echo ERROR: EDS dataset verification failed. Check the path %DATA%
    pause
    exit /b 1
)

echo.
echo =========================================================================
echo  STEP 1 — Generating labeled annotations for all 3 domains
echo =========================================================================

%PYTHON% data\prepare_labeled_annotations.py --dataset eds --data_root "%DATA%" --domain domain1 --out outputs\eds\domain1\labeled_annotations.json
if %errorlevel% neq 0 ( echo ERROR in annotations domain1 & pause & exit /b 1 )

%PYTHON% data\prepare_labeled_annotations.py --dataset eds --data_root "%DATA%" --domain domain2 --out outputs\eds\domain2\labeled_annotations.json
if %errorlevel% neq 0 ( echo ERROR in annotations domain2 & pause & exit /b 1 )

%PYTHON% data\prepare_labeled_annotations.py --dataset eds --data_root "%DATA%" --domain domain3 --out outputs\eds\domain3\labeled_annotations.json
if %errorlevel% neq 0 ( echo ERROR in annotations domain3 & pause & exit /b 1 )

echo.
echo =========================================================================
echo  STEP 2 — Stage 1: BYOL pretraining (one per source domain)
echo =========================================================================

echo.
echo [Stage 1] Domain 1 ...
%PYTHON% main.py --stage 1 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain1\stage1
if %errorlevel% neq 0 ( echo ERROR in Stage 1 domain1 & pause & exit /b 1 )

echo.
echo [Stage 1] Domain 2 ...
%PYTHON% main.py --stage 1 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain2\stage1
if %errorlevel% neq 0 ( echo ERROR in Stage 1 domain2 & pause & exit /b 1 )

echo.
echo [Stage 1] Domain 3 ...
%PYTHON% main.py --stage 1 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain3\stage1
if %errorlevel% neq 0 ( echo ERROR in Stage 1 domain3 & pause & exit /b 1 )

echo.
echo =========================================================================
echo  STEP 3 — Stage 2: source domain pseudo-labels
echo =========================================================================

echo.
echo [Stage 2] Domain 1 source pseudo-labels ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain1\stage1 --stage2_dir outputs\eds\domain1\stage2
if %errorlevel% neq 0 ( echo ERROR in Stage 2 domain1 source & pause & exit /b 1 )

echo.
echo [Stage 2] Domain 2 source pseudo-labels ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain2\stage1 --stage2_dir outputs\eds\domain2\stage2
if %errorlevel% neq 0 ( echo ERROR in Stage 2 domain2 source & pause & exit /b 1 )

echo.
echo [Stage 2] Domain 3 source pseudo-labels ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain3\stage1 --stage2_dir outputs\eds\domain3\stage2
if %errorlevel% neq 0 ( echo ERROR in Stage 2 domain3 source & pause & exit /b 1 )

echo.
echo =========================================================================
echo  STEP 4 — Stage 2: target domain proposals for cross-domain eval
echo =========================================================================

echo.
echo [Stage 2] D2 proposals using D1 encoder (for D1 to D2 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain1\stage1 --stage2_dir outputs\eds\domain2\stage2_from_d1
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo [Stage 2] D3 proposals using D1 encoder (for D1 to D3 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain1\stage1 --stage2_dir outputs\eds\domain3\stage2_from_d1
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo [Stage 2] D1 proposals using D2 encoder (for D2 to D1 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain2\stage1 --stage2_dir outputs\eds\domain1\stage2_from_d2
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo [Stage 2] D3 proposals using D2 encoder (for D2 to D3 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain2\stage1 --stage2_dir outputs\eds\domain3\stage2_from_d2
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo [Stage 2] D1 proposals using D3 encoder (for D3 to D1 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain3\stage1 --stage2_dir outputs\eds\domain1\stage2_from_d3
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo [Stage 2] D2 proposals using D3 encoder (for D3 to D2 eval) ...
%PYTHON% main.py --stage 2 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain3\stage1 --stage2_dir outputs\eds\domain2\stage2_from_d3
if %errorlevel% neq 0 ( echo ERROR & pause & exit /b 1 )

echo.
echo =========================================================================
echo  STEP 5 — Stage 3: all 6 cross-domain pairs (Table 4)
echo =========================================================================

echo.
echo [Stage 3] D1 to D2 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain1\stage1 --stage2_dir outputs\eds\domain1\stage2 --stage3_dir outputs\eds\d1_to_d2\stage3 --labeled_annotations outputs\eds\domain1\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain2\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain2\stage2_from_d1\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D1 to D2 & pause & exit /b 1 )

echo.
echo [Stage 3] D1 to D3 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain1 --stage1_dir outputs\eds\domain1\stage1 --stage2_dir outputs\eds\domain1\stage2 --stage3_dir outputs\eds\d1_to_d3\stage3 --labeled_annotations outputs\eds\domain1\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain3\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain3\stage2_from_d1\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D1 to D3 & pause & exit /b 1 )

echo.
echo [Stage 3] D2 to D1 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain2\stage1 --stage2_dir outputs\eds\domain2\stage2 --stage3_dir outputs\eds\d2_to_d1\stage3 --labeled_annotations outputs\eds\domain2\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain1\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain1\stage2_from_d2\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D2 to D1 & pause & exit /b 1 )

echo.
echo [Stage 3] D2 to D3 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain2 --stage1_dir outputs\eds\domain2\stage1 --stage2_dir outputs\eds\domain2\stage2 --stage3_dir outputs\eds\d2_to_d3\stage3 --labeled_annotations outputs\eds\domain2\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain3\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain3\stage2_from_d2\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D2 to D3 & pause & exit /b 1 )

echo.
echo [Stage 3] D3 to D1 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain3\stage1 --stage2_dir outputs\eds\domain3\stage2 --stage3_dir outputs\eds\d3_to_d1\stage3 --labeled_annotations outputs\eds\domain3\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain1\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain1\stage2_from_d3\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D3 to D1 & pause & exit /b 1 )

echo.
echo [Stage 3] D3 to D2 ...
%PYTHON% main.py --stage 3 --dataset eds --data_root "%DATA%" --domain domain3 --stage1_dir outputs\eds\domain3\stage1 --stage2_dir outputs\eds\domain3\stage2 --stage3_dir outputs\eds\d3_to_d2\stage3 --labeled_annotations outputs\eds\domain3\labeled_annotations.json --eval_labeled_annotations outputs\eds\domain2\labeled_annotations.json --eval_pseudo_labels outputs\eds\domain2\stage2_from_d3\pseudo_labels.json --no_resume
if %errorlevel% neq 0 ( echo ERROR in Stage 3 D3 to D2 & pause & exit /b 1 )

echo.
echo =========================================================================
echo  ALL DONE — Check outputs\eds\ for results
echo  Results per pair: outputs\eds\d1_to_d2\stage3\eval_results.json etc.
echo =========================================================================
pause
