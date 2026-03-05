import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="PlantaOS: Physical AI Brain for Built Environments")
    parser.add_argument("command", choices=["dashboard", "simulate", "train", "optimize"])
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--model", type=str, default="c3_lbm")
    parser.add_argument("--epochs", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        print("Starting PlantaOS Live Digital Twin (DT-02) Dashboard...")
        try:
            from ui import create_app
            app = create_app()
            app.run_server(debug=True, host='0.0.0.0', port=8050)
        except ImportError as e:
            print(f"Failed to load UI components: {e}")
            
    elif args.command == "simulate":
        print(f"Running Agent-Based Scenarios: {args.scenarios}")
        try:
            from scenarios import ScenarioMatrix
            matrix = ScenarioMatrix()
            matrix.run_sc01_morning_preheat()
            print("Simulations complete. Results ready for export.")
        except ImportError as e:
            print(f"Simulation failed: {e}")
            
    elif args.command == "train":
        print(f"Initializing LBM/LSTM Training on Deucalion (Model: {args.model}, Epochs: {args.epochs})")
        
    elif args.command == "optimize":
        print("Running Metaheuristic Optimization (NSGA-III / PSO) over F-Field...")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    main()
