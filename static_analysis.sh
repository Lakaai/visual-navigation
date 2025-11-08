#!/bin/bash

cppcheck --language=c++ --std=c++17 --platform=native --enable=all --suppress=missingIncludeSystem src test/src