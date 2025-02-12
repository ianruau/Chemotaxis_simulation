#!/usr/bin/env bash
black $1
isort $1
autopep8 --in-place --aggressive $1
