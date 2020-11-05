echo off
setlocal enabledelayedexpansion

set ROOTDIR=%~dp0\..\..\

bazel clean

for /f %%i in ('dir /a:d /b %ROOTDIR%\bazel-*') do rd /q %%i
rd /s /q %ROOTDIR%\out