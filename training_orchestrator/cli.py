"""
Command Line Interface for GAELP Training Orchestrator

Provides easy access to training orchestrator functionality from the command line.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .config import (
    TrainingOrchestratorConfig, 
    DEVELOPMENT_CONFIG, 
    STAGING_CONFIG, 
    PRODUCTION_CONFIG,
    QUICK_TEST_CONFIG
)
from .core import TrainingOrchestrator, TrainingConfiguration


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training_orchestrator.log')
        ]
    )


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="GAELP Training Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  gaelp-train --config quick-test --log-level DEBUG
  
  # Run development training
  gaelp-train --config development --experiment-name "my_experiment"
  
  # Run with custom configuration
  gaelp-train --config-file config.yaml --environment production
  
  # Validate configuration
  gaelp-train --validate-config --config staging
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config",
        choices=["development", "staging", "production", "quick-test"],
        default="development",
        help="Predefined configuration to use"
    )
    config_group.add_argument(
        "--config-file",
        type=Path,
        help="Path to custom configuration file"
    )
    
    # Training options
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the training experiment"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        help="Override environment setting"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running training"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["simulation", "historical", "real", "scaled"],
        help="Specific phases to run (default: all phases)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./training_output"),
        help="Directory for training outputs"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory for checkpoints"
    )
    
    return parser


def load_configuration(args) -> TrainingOrchestratorConfig:
    """Load configuration based on arguments"""
    
    if args.config_file:
        # Load from file (implementation depends on file format)
        raise NotImplementedError("Configuration file loading not yet implemented")
    
    # Load predefined configuration
    config_map = {
        "development": DEVELOPMENT_CONFIG,
        "staging": STAGING_CONFIG,
        "production": PRODUCTION_CONFIG,
        "quick-test": QUICK_TEST_CONFIG
    }
    
    config = config_map[args.config]
    
    # Apply overrides from arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if args.environment:
        config.environment = args.environment
    
    if args.log_level:
        config.monitoring.log_level = args.log_level
    
    return config


def validate_configuration(config: TrainingOrchestratorConfig) -> bool:
    """Validate configuration and report issues"""
    
    print("Validating configuration...")
    
    issues = config.validate()
    
    if not issues:
        print("✅ Configuration is valid")
        return True
    
    print("❌ Configuration validation failed:")
    for issue in issues:
        print(f"  - {issue}")
    
    return False


def print_configuration_summary(config: TrainingOrchestratorConfig):
    """Print a summary of the configuration"""
    
    print(f"\nConfiguration Summary:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Environment: {config.environment}")
    print(f"  Random Seed: {config.random_seed}")
    
    print(f"\nPhase Settings:")
    print(f"  Simulation Episodes: {config.phases.simulation_episodes}")
    print(f"  Historical Validation Episodes: {config.phases.historical_validation_episodes}")
    print(f"  Real Testing Episodes: {config.phases.real_testing_episodes}")
    print(f"  Scaled Deployment Episodes: {config.phases.scaled_deployment_episodes}")
    
    print(f"\nBudget Settings:")
    print(f"  Real Testing Daily Limit: ${config.budget.real_testing_daily_limit}")
    print(f"  Scaled Deployment Daily Limit: ${config.budget.scaled_deployment_daily_limit}")
    
    print(f"\nSafety Settings:")
    print(f"  Content Safety Threshold: {config.safety.content_safety_threshold}")
    print(f"  Brand Safety Threshold: {config.safety.brand_safety_threshold}")
    print(f"  Require Human Approval: {config.safety.require_human_approval_real}")
    print(f"  Max Violations Per Day: {config.safety.max_violations_per_day}")
    
    print(f"\nMonitoring Settings:")
    print(f"  Log Level: {config.monitoring.log_level}")
    print(f"  Checkpoint Interval: {config.monitoring.checkpoint_interval}")


async def run_training_orchestrator(config: TrainingOrchestratorConfig, args):
    """Run the training orchestrator"""
    
    print("🚀 Starting GAELP Training Orchestrator")
    
    # Convert to legacy configuration format
    legacy_config = TrainingConfiguration(**config.to_legacy_config())
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(legacy_config)
    
    print("❗ Note: This CLI currently requires you to provide agent and environment implementations")
    print("❗ Please see example_training_run.py for a complete implementation example")
    print("❗ The CLI will be enhanced in future versions to support plugin-based agents/environments")
    
    # For now, just show the orchestrator was created successfully
    print(f"✅ Training Orchestrator initialized successfully")
    print(f"   Experiment ID: {legacy_config.experiment_id}")
    print(f"   Current State: {orchestrator.get_state().value}")
    
    # Get some basic metrics
    metrics = orchestrator.get_metrics()
    print(f"   Metrics: {metrics}")
    
    print("\n📋 To run actual training, please:")
    print("   1. Implement your agent class")
    print("   2. Implement your environment classes")
    print("   3. Use the Python API directly")
    print("   4. See example_training_run.py for reference")


async def main_async(args):
    """Main async function"""
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    try:
        config = load_configuration(args)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Print configuration summary
    print_configuration_summary(config)
    
    # Validate configuration
    if not validate_configuration(config):
        return 1
    
    # If only validating, exit here
    if args.validate_config:
        return 0
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print("✅ Dry run completed successfully")
        return 0
    
    # Run training orchestrator
    try:
        await run_training_orchestrator(config, args)
        return 0
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Run async main
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())