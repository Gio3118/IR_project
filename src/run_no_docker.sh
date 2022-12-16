#!/bin/bash

# This script is used to run the application without docker - search service won't work
chmod +x ./webui/run.sh
chmod +x ./scraping-service/run.sh
chmod +x ./sentiment-analysis-service/run.sh

./webui/run.sh & ./scraping-service/run.sh & ./sentiment-analysis-service/run.sh

