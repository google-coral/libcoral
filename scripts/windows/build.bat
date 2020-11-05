echo off
setlocal enabledelayedexpansion

if not defined PYTHON ( set PYTHON=python )

set BAZEL_CMD=bazel
if defined BAZEL_OUTPUT_BASE (
    set BAZEL_CMD=%BAZEL_CMD% --output_base=%BAZEL_OUTPUT_BASE%
)

set BAZEL_INFO_FLAGS=^
--experimental_repo_remote_exec

set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
call "%BAZEL_VC%\Auxiliary\Build\vcvars64.bat"
type NUL >>BUILD

for /f %%i in ('%BAZEL_CMD% info %BAZEL_INFO_FLAGS% output_base') do set "BAZEL_OUTPUT_BASE=%%i"
for /f %%i in ('%BAZEL_CMD% info %BAZEL_INFO_FLAGS% output_path') do set "BAZEL_OUTPUT_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(str(sys.version_info.major)+str(sys.version_info.minor))"') do set "PY3_VER=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.executable)"') do set "PYTHON_BIN_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.base_prefix)"') do set "PYTHON_LIB_PATH=%%i\Lib"

set BAZEL_OUTPUT_PATH=%BAZEL_OUTPUT_PATH:/=\%
set BAZEL_OUTPUT_BASE=%BAZEL_OUTPUT_BASE:/=\%
set CPU=x64_windows
set COMPILATION_MODE=opt
set LIBEDGETPU_VERSION=direct

set ROOTDIR=%~dp0\..\..\
set BAZEL_OUT_DIR=%BAZEL_OUTPUT_PATH%\%CPU%-%COMPILATION_MODE%\bin
set PYBIND_OUT_DIR=%ROOTDIR%\edgetpu\pybind
set TOOLS_OUT_DIR=%ROOTDIR%\out\%CPU%\tools
set EXAMPLES_OUT_DIR=%ROOTDIR%\out\%CPU%\examples
set TESTS_OUT_DIR=%ROOTDIR%\out\%CPU%\tests
set BENCHMARKS_OUT_DIR=%ROOTDIR%\out\%CPU%\benchmarks
set TFLITE_WRAPPER_OUT_DIR=%ROOTDIR%\tflite_runtime

set TFLITE_WRAPPER_NAME=_pywrap_tensorflow_interpreter_wrapper.cp%PY3_VER%-win_amd64.pyd
set PYBIND_WRAPPER_NAME=_pywrap_coral.cp%PY3_VER%-win_amd64.pyd

set TFLITE_WRAPPER_PATH=%TFLITE_WRAPPER_OUT_DIR%\%TFLITE_WRAPPER_NAME%
set PYBIND_WRAPPER_PATH=%PYBIND_OUT_DIR%\%PYBIND_WRAPPER_NAME%

:PROCESSARGS
set ARG=%1
if defined ARG (
    if "%ARG%"=="/DBG" (
        set COMPILATION_MODE=dbg
    )
    shift
    goto PROCESSARGS
)

for /f "tokens=3" %%i in ('type %ROOTDIR%\WORKSPACE ^| findstr /C:"TENSORFLOW_COMMIT ="') do set "TENSORFLOW_COMMIT=%%i"
set BAZEL_BUILD_FLAGS= ^
--compilation_mode=%COMPILATION_MODE% ^
--copt=/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ^
--copt=/D_HAS_DEPRECATED_RESULT_OF ^
--linkopt=/DEFAULTLIB:%BAZEL_OUTPUT_BASE%\external\libusb\root\MS64\dll\libusb-1.0.lib ^
--define darwinn_portable=1 ^
--experimental_repo_remote_exec ^
--copt=/std:c++latest ^
--embed_label=%TENSORFLOW_COMMIT% ^
--stamp
set BAZEL_QUERY_FLAGS=^
--experimental_repo_remote_exec

rem Tests
for /F "tokens=* USEBACKQ" %%g in (`%BAZEL_CMD% query %BAZEL_QUERY_FLAGS% "kind(cc_.*test, //coral/...) except //coral/dmabuf:all"`) do (set "tests=!tests! %%g")
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% %tests% || goto exit
for /F %%i in ('dir /a:-d /s /b %BAZEL_OUT_DIR%\*_test.exe') do (
    set out_dir="%%~dpi"
    set out_dir=!out_dir:%BAZEL_OUT_DIR%\=!
    set out_dir=%TESTS_OUT_DIR%\!out_dir!
    if not exist !out_dir! md !out_dir!
    copy %%i !out_dir! >NUL
)

rem Benchmarks
for /F "tokens=* USEBACKQ" %%g in (`%BAZEL_CMD% query %BAZEL_QUERY_FLAGS% "kind(cc_binary, //coral/...)"`) do (echo %%g | findstr benchmark >NUL && set "benchmarks=!benchmarks! %%g")
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% %benchmarks% || goto exit
for /F %%i in ('dir /a:-d /s /b %BAZEL_OUT_DIR%\*_benchmark.exe') do (
    set out_dir="%%~dpi"
    set out_dir=!out_dir:%BAZEL_OUT_DIR%\=!
    set out_dir=%BENCHMARKS_OUT_DIR%\!out_dir!
    if not exist !out_dir! md !out_dir!
    copy %%i !out_dir! >NUL
)

rem Tools
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% ^
    //coral/tools:join_tflite_models ^
    //coral/tools:multiple_tpus_performance_analysis ^
    //coral/tools:model_pipelining_performance_analysis || goto exit
if not exist %TOOLS_OUT_DIR% md %TOOLS_OUT_DIR%
copy %BAZEL_OUT_DIR%\coral\tools\join_tflite_models.exe %TOOLS_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\coral\tools\multiple_tpus_performance_analysis.exe %TOOLS_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\coral\tools\model_pipelining_performance_analysis.exe %TOOLS_OUT_DIR% >NUL

rem Examples
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% ^
    //coral/examples:two_models_one_tpu ^
    //coral/examples:two_models_two_tpus_threaded ^
    //coral/examples:classify_image ^
    //coral/examples:model_pipelining || goto exit
if not exist %EXAMPLES_OUT_DIR% md %EXAMPLES_OUT_DIR%
copy %BAZEL_OUT_DIR%\coral\examples\two_models_one_tpu.exe %EXAMPLES_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\coral\examples\two_models_two_tpus_threaded.exe %EXAMPLES_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\coral\examples\classify_image.exe %EXAMPLES_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\coral\examples\model_pipelining.exe %EXAMPLES_OUT_DIR% >NUL

:exit
exit /b %ERRORLEVEL%