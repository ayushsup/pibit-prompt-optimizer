from src.optimizer.loop import OptimizerLoop

def main():
    try:
        # Initialize and run the optimizer with the base configuration
        optimizer = OptimizerLoop(config_path="config/base_config.yaml")
        optimizer.run()
    except KeyboardInterrupt:
        print("\nRun interrupted by user. State is safely persisted in SQLite.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()