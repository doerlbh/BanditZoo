#!/bin/bash
coverage run --source=. -m nose
rm coverage.svg
coverage-badge -o coverage.svg