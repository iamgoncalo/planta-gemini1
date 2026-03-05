# PLANTAOS — Physical AI Brain for Built Environments

**Deployment:** HORSE CFT, Aveiro, Portugal (~950 sqm)
**Grant:** FCT 2025.00020.AIVLAB.DEUCALION
**Author:** Goncalo Melo

Master implementation of the Architecture of Freedom Intelligence (F = P/D).

## 1. Installation & Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## 2. Run Automated Tests
pytest tests/ --tb=short -v

## 3. How to Run PlantaOS
Dashboard: python3 plantaos.py dashboard
Simulate: python3 plantaos.py simulate --scenarios all
Train: python3 plantaos.py train --model c3_lbm --epochs 500
