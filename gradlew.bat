@ECHO OFF
SETLOCAL
SET GRADLE_CMD=gradle
WHERE %GRADLE_CMD% >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
  ECHO Gradle is required to build this project.
  EXIT /B 1
)
%GRADLE_CMD% %*
