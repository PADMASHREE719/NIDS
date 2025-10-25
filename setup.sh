#!/usr/bin/env bash
set -e
# ubuntu example
sudo apt update
sudo apt install -y openjdk-11-jre-headless build-essential libpcap-dev python3-venv git
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Put dataset PCAPs/CSVs into data/raw/ as described in README."
