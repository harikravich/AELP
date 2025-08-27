#!/usr/bin/env python3
"""
Model Versioning and Experiment Tracking System for GAELP.
Provides Git-based model tracking, experiment metadata storage, 
model performance history, A/B test results tracking, and rollback capabilities.
Integrates with Weights & Biases for comprehensive MLOps.
"""

import os
import json
import git
import wandb
import hashlib
import shutil
import pickle
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

from wandb_tracking import GAELPWandbTracker, GAELPExperimentConfig

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model version status enum"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ExperimentType(Enum):
    """Experiment type enum"""
    TRAINING = "training"
    AB_TEST = "ab_test"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    HYPERPARAMETER_SWEEP = "hyperparameter_sweep"


@dataclass
class ModelMetadata:
    """Model version metadata"""
    model_id: str
    version: str
    git_commit: str
    git_branch: str
    timestamp: datetime
    status: ModelStatus
    experiment_id: str
    model_path: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    parent_version: Optional[str] = None
    description: str = ""
    author: str = ""


@dataclass
class ExperimentMetadata:
    """Experiment metadata"""
    experiment_id: str
    experiment_type: ExperimentType
    name: str
    description: str
    timestamp: datetime
    status: str
    config: Dict[str, Any]
    model_versions: List[str]
    metrics: Dict[str, Any]
    tags: List[str]
    wandb_run_id: Optional[str] = None
    author: str = ""
    duration_seconds: Optional[float] = None


@dataclass
class ABTestResult:
    """A/B test result data"""
    test_id: str
    experiment_id: str
    model_a_version: str
    model_b_version: str
    timestamp: datetime
    duration_seconds: float
    traffic_split: Dict[str, float]  # {"model_a": 0.5, "model_b": 0.5}
    metrics: Dict[str, Dict[str, float]]  # {"model_a": {...}, "model_b": {...}}
    statistical_significance: Dict[str, float]
    winner: Optional[str] = None
    confidence_level: float = 0.95


class ModelVersioningSystem:
    """
    Comprehensive model versioning and experiment tracking system.
    Provides Git-based version control, experiment management, and A/B testing.
    """
    
    def __init__(
        self,
        models_dir: str = "./models",
        experiments_dir: str = "./experiments",
        git_repo_path: str = ".",
        wandb_project: str = "gaelp-model-versioning",
        auto_git_commit: bool = True
    ):
        """
        Initialize the model versioning system
        
        Args:
            models_dir: Directory to store model files
            experiments_dir: Directory to store experiment metadata
            git_repo_path: Path to the git repository
            wandb_project: W&B project name for tracking
            auto_git_commit: Automatically commit model versions to git
        """
        self.models_dir = Path(models_dir)
        self.experiments_dir = Path(experiments_dir)
        self.git_repo_path = Path(git_repo_path)
        self.wandb_project = wandb_project
        self.auto_git_commit = auto_git_commit
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Initialize git repo if needed
        self._init_git_repo()
        
        # Storage for metadata
        self.models_metadata: Dict[str, ModelMetadata] = {}
        self.experiments_metadata: Dict[str, ExperimentMetadata] = {}
        self.ab_tests: Dict[str, ABTestResult] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"Initialized ModelVersioningSystem with models_dir: {self.models_dir}")
    
    def _init_git_repo(self):
        """Initialize git repository if it doesn't exist"""
        try:
            self.repo = git.Repo(self.git_repo_path)
            logger.info("Connected to existing git repository")
        except git.InvalidGitRepositoryError:
            if self.auto_git_commit:
                self.repo = git.Repo.init(self.git_repo_path)
                logger.info("Initialized new git repository")
            else:
                self.repo = None
                logger.warning("No git repository found and auto_git_commit is disabled")
        except Exception as e:
            logger.warning(f"Git repository error: {e}")
            self.repo = None
    
    def _load_metadata(self):
        """Load existing metadata from disk"""
        try:
            # Load models metadata
            models_metadata_file = self.experiments_dir / "models_metadata.json"
            if models_metadata_file.exists():
                with open(models_metadata_file, 'r') as f:
                    data = json.load(f)
                    for model_id, metadata in data.items():
                        metadata['timestamp'] = datetime.fromisoformat(metadata['timestamp'])
                        metadata['status'] = ModelStatus(metadata['status'])
                        self.models_metadata[model_id] = ModelMetadata(**metadata)
            
            # Load experiments metadata
            experiments_metadata_file = self.experiments_dir / "experiments_metadata.json"
            if experiments_metadata_file.exists():
                with open(experiments_metadata_file, 'r') as f:
                    data = json.load(f)
                    for exp_id, metadata in data.items():
                        metadata['timestamp'] = datetime.fromisoformat(metadata['timestamp'])
                        metadata['experiment_type'] = ExperimentType(metadata['experiment_type'])
                        self.experiments_metadata[exp_id] = ExperimentMetadata(**metadata)
            
            # Load A/B tests
            ab_tests_file = self.experiments_dir / "ab_tests.json"
            if ab_tests_file.exists():
                with open(ab_tests_file, 'r') as f:
                    data = json.load(f)
                    for test_id, test_data in data.items():
                        test_data['timestamp'] = datetime.fromisoformat(test_data['timestamp'])
                        self.ab_tests[test_id] = ABTestResult(**test_data)
            
            logger.info(f"Loaded metadata: {len(self.models_metadata)} models, {len(self.experiments_metadata)} experiments")
            
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            # Save models metadata
            models_data = {}
            for model_id, metadata in self.models_metadata.items():
                data = asdict(metadata)
                data['timestamp'] = data['timestamp'].isoformat()
                data['status'] = data['status'].value
                models_data[model_id] = data
            
            with open(self.experiments_dir / "models_metadata.json", 'w') as f:
                json.dump(models_data, f, indent=2, default=str)
            
            # Save experiments metadata
            experiments_data = {}
            for exp_id, metadata in self.experiments_metadata.items():
                data = asdict(metadata)
                data['timestamp'] = data['timestamp'].isoformat()
                data['experiment_type'] = data['experiment_type'].value
                experiments_data[exp_id] = data
            
            with open(self.experiments_dir / "experiments_metadata.json", 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
            
            # Save A/B tests
            ab_tests_data = {}
            for test_id, test_result in self.ab_tests.items():
                data = asdict(test_result)
                data['timestamp'] = data['timestamp'].isoformat()
                ab_tests_data[test_id] = data
            
            with open(self.experiments_dir / "ab_tests.json", 'w') as f:
                json.dump(ab_tests_data, f, indent=2, default=str)
            
            logger.info("Saved metadata to disk")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _get_git_info(self) -> Tuple[str, str]:
        """Get current git commit and branch"""
        if not self.repo:
            return "unknown", "unknown"
        
        try:
            commit = self.repo.head.commit.hexsha
            branch = self.repo.active_branch.name
            return commit, branch
        except Exception as e:
            logger.warning(f"Failed to get git info: {e}")
            return "unknown", "unknown"
    
    def _generate_model_id(self, model_name: str, config: Dict[str, Any]) -> str:
        """Generate unique model ID based on name and config"""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}_{config_hash}"
    
    def _generate_version(self, model_id: str) -> str:
        """Generate version number for model"""
        existing_versions = [
            m.version for m in self.models_metadata.values() 
            if m.model_id.startswith(model_id.split('_')[0])
        ]
        
        if not existing_versions:
            return "v1.0.0"
        
        # Simple incremental versioning
        max_version = max([int(v.split('.')[1]) for v in existing_versions if v.startswith('v1.')])
        return f"v1.{max_version + 1}.0"
    
    def save_model_version(
        self,
        model_obj: Any,
        model_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        experiment_id: str,
        tags: List[str] = None,
        description: str = "",
        status: ModelStatus = ModelStatus.TRAINING,
        parent_version: str = None
    ) -> str:
        """
        Save a new model version with metadata tracking
        
        Args:
            model_obj: The trained model object (PyTorch, TensorFlow, etc.)
            model_name: Name of the model
            config: Model configuration and hyperparameters
            metrics: Performance metrics
            experiment_id: Associated experiment ID
            tags: Tags for the model
            description: Model description
            status: Model status
            parent_version: Parent model version for lineage tracking
        
        Returns:
            Model version ID
        """
        try:
            # Generate model ID and version
            model_id = self._generate_model_id(model_name, config)
            version = self._generate_version(model_id)
            
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model file
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_obj, f)
            
            # Save config
            with open(model_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            # Save metrics
            with open(model_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Get git info
            git_commit, git_branch = self._get_git_info()
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                version=version,
                git_commit=git_commit,
                git_branch=git_branch,
                timestamp=datetime.now(timezone.utc),
                status=status,
                experiment_id=experiment_id,
                model_path=str(model_path),
                config=config,
                metrics=metrics,
                tags=tags or [],
                parent_version=parent_version,
                description=description,
                author=os.environ.get('USER', 'unknown')
            )
            
            # Store metadata
            self.models_metadata[model_id] = metadata
            
            # Add model to experiment's model list
            if experiment_id in self.experiments_metadata:
                exp_metadata = self.experiments_metadata[experiment_id]
                if model_id not in exp_metadata.model_versions:
                    exp_metadata.model_versions.append(model_id)
            
            self._save_metadata()
            
            # Commit to git if enabled
            if self.auto_git_commit and self.repo:
                try:
                    self.repo.git.add(str(model_dir))
                    self.repo.git.add(str(self.experiments_dir / "models_metadata.json"))
                    self.repo.index.commit(f"Add model version {model_id} {version}")
                    logger.info(f"Committed model version {model_id} to git")
                except Exception as e:
                    logger.warning(f"Failed to commit to git: {e}")
            
            # Log to W&B if experiment exists
            if experiment_id in self.experiments_metadata:
                exp_metadata = self.experiments_metadata[experiment_id]
                if exp_metadata.wandb_run_id:
                    try:
                        # Create artifact
                        artifact = wandb.Artifact(
                            name=f"{model_name}_{version}",
                            type="model",
                            description=description or f"Model {model_id} version {version}",
                            metadata={
                                "model_id": model_id,
                                "version": version,
                                "metrics": metrics,
                                "config": config
                            }
                        )
                        artifact.add_dir(str(model_dir))
                        
                        # Log artifact
                        run = wandb.init(
                            project=self.wandb_project,
                            id=exp_metadata.wandb_run_id,
                            resume="allow"
                        )
                        run.log_artifact(artifact)
                        wandb.finish()
                        
                    except Exception as e:
                        logger.warning(f"Failed to log model to W&B: {e}")
            
            logger.info(f"Saved model version {model_id} {version}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to save model version: {e}")
            raise
    
    def track_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        config: Dict[str, Any],
        description: str = "",
        tags: List[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new experiment
        
        Args:
            name: Experiment name
            experiment_type: Type of experiment
            config: Experiment configuration
            description: Experiment description
            tags: Tags for the experiment
            wandb_config: W&B specific configuration
        
        Returns:
            Experiment ID
        """
        try:
            # Generate experiment ID
            experiment_id = f"{experiment_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize W&B tracking
            wandb_run_id = None
            if wandb_config is not None:
                try:
                    tracker = GAELPWandbTracker(
                        project_name=self.wandb_project,
                        experiment_name=f"{name}_{experiment_id}",
                        config=wandb_config,
                        tags=tags
                    )
                    wandb_run_id = tracker.run.id if tracker.run else None
                except Exception as e:
                    logger.warning(f"Failed to initialize W&B tracking: {e}")
            
            # Create experiment metadata
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                name=name,
                description=description,
                timestamp=datetime.now(timezone.utc),
                status="running",
                config=config,
                model_versions=[],
                metrics={},
                tags=tags or [],
                wandb_run_id=wandb_run_id,
                author=os.environ.get('USER', 'unknown')
            )
            
            # Store metadata
            self.experiments_metadata[experiment_id] = metadata
            self._save_metadata()
            
            logger.info(f"Started tracking experiment {experiment_id}: {name}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start experiment tracking: {e}")
            raise
    
    def update_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, Any] = None,
        status: str = None,
        model_versions: List[str] = None
    ):
        """Update experiment with new metrics or status"""
        try:
            if experiment_id not in self.experiments_metadata:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            metadata = self.experiments_metadata[experiment_id]
            
            if metrics:
                metadata.metrics.update(metrics)
            
            if status:
                metadata.status = status
                if status == "completed":
                    start_time = metadata.timestamp
                    metadata.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            if model_versions:
                # Only add model versions that aren't already in the list
                for model_id in model_versions:
                    if model_id not in metadata.model_versions:
                        metadata.model_versions.append(model_id)
            
            self._save_metadata()
            logger.info(f"Updated experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to update experiment: {e}")
            raise
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            version1: First model ID to compare
            version2: Second model ID to compare
            metrics: Specific metrics to compare (all if None)
        
        Returns:
            Comparison results
        """
        try:
            if version1 not in self.models_metadata:
                raise ValueError(f"Model version {version1} not found")
            
            if version2 not in self.models_metadata:
                raise ValueError(f"Model version {version2} not found")
            
            model1 = self.models_metadata[version1]
            model2 = self.models_metadata[version2]
            
            # Compare metrics
            comparison = {
                "version1": {
                    "model_id": version1,
                    "version": model1.version,
                    "timestamp": model1.timestamp.isoformat(),
                    "status": model1.status.value,
                    "metrics": model1.metrics
                },
                "version2": {
                    "model_id": version2,
                    "version": model2.version,
                    "timestamp": model2.timestamp.isoformat(),
                    "status": model2.status.value,
                    "metrics": model2.metrics
                },
                "differences": {}
            }
            
            # Calculate metric differences
            metrics_to_compare = metrics or set(model1.metrics.keys()) | set(model2.metrics.keys())
            
            for metric in metrics_to_compare:
                val1 = model1.metrics.get(metric, 0)
                val2 = model2.metrics.get(metric, 0)
                
                comparison["differences"][metric] = {
                    "absolute_diff": val2 - val1,
                    "relative_diff": ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf'),
                    "winner": "version2" if val2 > val1 else "version1" if val1 > val2 else "tie"
                }
            
            # Overall winner based on primary metric (assuming 'roas' or first metric)
            primary_metric = metrics[0] if metrics else list(comparison["differences"].keys())[0]
            comparison["overall_winner"] = comparison["differences"][primary_metric]["winner"]
            
            logger.info(f"Compared versions {version1} vs {version2}")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    def run_ab_test(
        self,
        test_name: str,
        model_a_version: str,
        model_b_version: str,
        traffic_split: Dict[str, float] = None,
        duration_hours: float = 24.0,
        metrics_callback: callable = None
    ) -> str:
        """
        Run an A/B test between two model versions
        
        Args:
            test_name: Name of the A/B test
            model_a_version: First model version ID
            model_b_version: Second model version ID
            traffic_split: Traffic split dictionary {"model_a": 0.5, "model_b": 0.5}
            duration_hours: Test duration in hours
            metrics_callback: Function to collect metrics during test
        
        Returns:
            A/B test ID
        """
        try:
            if model_a_version not in self.models_metadata:
                raise ValueError(f"Model version {model_a_version} not found")
            
            if model_b_version not in self.models_metadata:
                raise ValueError(f"Model version {model_b_version} not found")
            
            # Generate test ID
            test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Default traffic split
            if traffic_split is None:
                traffic_split = {"model_a": 0.5, "model_b": 0.5}
            
            # Create experiment for the A/B test
            experiment_id = self.track_experiment(
                name=f"AB_Test_{test_name}",
                experiment_type=ExperimentType.AB_TEST,
                config={
                    "test_id": test_id,
                    "model_a_version": model_a_version,
                    "model_b_version": model_b_version,
                    "traffic_split": traffic_split,
                    "duration_hours": duration_hours
                },
                description=f"A/B test comparing {model_a_version} vs {model_b_version}",
                tags=["ab_test", test_name]
            )
            
            # TODO: Implement actual A/B test execution logic
            # This would involve:
            # 1. Deploying both models
            # 2. Splitting traffic
            # 3. Collecting metrics
            # 4. Statistical analysis
            
            # For now, simulate with dummy metrics
            dummy_metrics = {
                "model_a": {
                    "roas": np.random.normal(3.2, 0.5),
                    "ctr": np.random.normal(0.08, 0.02),
                    "conversion_rate": np.random.normal(0.12, 0.03),
                    "total_revenue": np.random.normal(10000, 2000)
                },
                "model_b": {
                    "roas": np.random.normal(3.4, 0.5),
                    "ctr": np.random.normal(0.09, 0.02),
                    "conversion_rate": np.random.normal(0.13, 0.03),
                    "total_revenue": np.random.normal(11000, 2000)
                }
            }
            
            # Calculate statistical significance (simplified)
            significance = {}
            for metric in dummy_metrics["model_a"].keys():
                val_a = dummy_metrics["model_a"][metric]
                val_b = dummy_metrics["model_b"][metric]
                # Simplified p-value calculation (in practice, use proper statistical tests)
                significance[metric] = abs(val_b - val_a) / max(val_a, val_b) * 100
            
            # Determine winner
            primary_metric = "roas"
            winner = "model_b" if dummy_metrics["model_b"][primary_metric] > dummy_metrics["model_a"][primary_metric] else "model_a"
            
            # Create A/B test result
            ab_test_result = ABTestResult(
                test_id=test_id,
                experiment_id=experiment_id,
                model_a_version=model_a_version,
                model_b_version=model_b_version,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration_hours * 3600,
                traffic_split=traffic_split,
                metrics=dummy_metrics,
                statistical_significance=significance,
                winner=winner,
                confidence_level=0.95
            )
            
            # Store result
            self.ab_tests[test_id] = ab_test_result
            
            # Update experiment
            self.update_experiment(
                experiment_id,
                metrics={
                    "ab_test_winner": winner,
                    "confidence_level": 0.95,
                    "primary_metric_improvement": significance[primary_metric]
                },
                status="completed"
            )
            
            self._save_metadata()
            
            logger.info(f"Completed A/B test {test_id}: {winner} wins")
            return test_id
            
        except Exception as e:
            logger.error(f"Failed to run A/B test: {e}")
            raise
    
    def rollback(self, target_version: str, reason: str = "") -> bool:
        """
        Rollback to a previous model version
        
        Args:
            target_version: Model version ID to rollback to
            reason: Reason for rollback
        
        Returns:
            Success status
        """
        try:
            if target_version not in self.models_metadata:
                raise ValueError(f"Target version {target_version} not found")
            
            target_metadata = self.models_metadata[target_version]
            
            # Validate target version is suitable for rollback
            if target_metadata.status not in [ModelStatus.PRODUCTION, ModelStatus.VALIDATION]:
                logger.warning(f"Rolling back to version with status: {target_metadata.status}")
            
            # Create rollback experiment
            experiment_id = self.track_experiment(
                name=f"Rollback_to_{target_version}",
                experiment_type=ExperimentType.TRAINING,  # Could add ROLLBACK type
                config={
                    "rollback_target": target_version,
                    "reason": reason,
                    "original_model_path": target_metadata.model_path
                },
                description=f"Rollback to version {target_version}: {reason}",
                tags=["rollback", target_version]
            )
            
            # Copy model files to current location (implementation depends on deployment strategy)
            current_model_dir = self.models_dir / "current"
            current_model_dir.mkdir(exist_ok=True)
            
            # Copy model files
            target_model_dir = Path(target_metadata.model_path).parent
            for file_path in target_model_dir.glob("*"):
                shutil.copy2(file_path, current_model_dir)
            
            # Update target version status to production
            target_metadata.status = ModelStatus.PRODUCTION
            
            # Commit rollback to git
            if self.auto_git_commit and self.repo:
                try:
                    self.repo.git.add(str(current_model_dir))
                    self.repo.index.commit(f"Rollback to {target_version}: {reason}")
                    logger.info(f"Committed rollback to git")
                except Exception as e:
                    logger.warning(f"Failed to commit rollback: {e}")
            
            # Update experiment
            self.update_experiment(
                experiment_id,
                metrics={"rollback_completed": True},
                status="completed",
                model_versions=[target_version]
            )
            
            self._save_metadata()
            
            logger.info(f"Successfully rolled back to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to {target_version}: {e}")
            return False
    
    def get_model_history(self, model_name_prefix: str = "") -> List[ModelMetadata]:
        """Get model version history"""
        models = [
            metadata for metadata in self.models_metadata.values()
            if model_name_prefix in metadata.model_id
        ]
        return sorted(models, key=lambda x: x.timestamp, reverse=True)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results"""
        if experiment_id not in self.experiments_metadata:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_metadata = self.experiments_metadata[experiment_id]
        
        # Get associated models
        models = [
            self.models_metadata[model_id] 
            for model_id in exp_metadata.model_versions
            if model_id in self.models_metadata
        ]
        
        # Get A/B tests
        ab_tests = [
            test for test in self.ab_tests.values()
            if test.experiment_id == experiment_id
        ]
        
        return {
            "experiment": asdict(exp_metadata),
            "models": [asdict(model) for model in models],
            "ab_tests": [asdict(test) for test in ab_tests],
            "summary": {
                "total_models": len(models),
                "total_ab_tests": len(ab_tests),
                "duration_hours": exp_metadata.duration_seconds / 3600 if exp_metadata.duration_seconds else None
            }
        }
    
    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models currently in production"""
        return [
            metadata for metadata in self.models_metadata.values()
            if metadata.status == ModelStatus.PRODUCTION
        ]
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model lineage (parent-child relationships)"""
        if model_id not in self.models_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models_metadata[model_id]
        
        # Find parents
        parents = []
        current_parent = model.parent_version
        while current_parent and current_parent in self.models_metadata:
            parents.append(self.models_metadata[current_parent])
            current_parent = self.models_metadata[current_parent].parent_version
        
        # Find children
        children = [
            metadata for metadata in self.models_metadata.values()
            if metadata.parent_version == model_id
        ]
        
        return {
            "model": asdict(model),
            "parents": [asdict(p) for p in parents],
            "children": [asdict(c) for c in children],
            "depth": len(parents)
        }
    
    def export_experiment_report(self, experiment_id: str, output_path: str = None) -> str:
        """Export comprehensive experiment report"""
        try:
            results = self.get_experiment_results(experiment_id)
            
            if output_path is None:
                output_path = f"experiment_report_{experiment_id}.json"
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Exported experiment report to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export experiment report: {e}")
            raise
    
    def cleanup_old_versions(self, days_old: int = 30, keep_production: bool = True):
        """Clean up old model versions"""
        try:
            cutoff_date = datetime.now(timezone.utc) - datetime.timedelta(days=days_old)
            
            models_to_remove = []
            for model_id, metadata in self.models_metadata.items():
                if metadata.timestamp < cutoff_date:
                    if keep_production and metadata.status == ModelStatus.PRODUCTION:
                        continue
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                metadata = self.models_metadata[model_id]
                
                # Remove model files
                model_dir = Path(metadata.model_path).parent
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                # Remove from metadata
                del self.models_metadata[model_id]
                
                logger.info(f"Cleaned up old model version {model_id}")
            
            self._save_metadata()
            logger.info(f"Cleaned up {len(models_to_remove)} old model versions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            raise


def create_versioning_system(
    models_dir: str = "./models",
    experiments_dir: str = "./experiments",
    git_repo_path: str = ".",
    wandb_project: str = "gaelp-model-versioning"
) -> ModelVersioningSystem:
    """
    Factory function to create a ModelVersioningSystem instance
    
    Args:
        models_dir: Directory to store model files
        experiments_dir: Directory to store experiment metadata
        git_repo_path: Path to the git repository
        wandb_project: W&B project name for tracking
    
    Returns:
        Initialized ModelVersioningSystem
    """
    return ModelVersioningSystem(
        models_dir=models_dir,
        experiments_dir=experiments_dir,
        git_repo_path=git_repo_path,
        wandb_project=wandb_project
    )


# Example usage and integration functions
def integrate_with_gaelp_training():
    """Example of how to integrate with GAELP training pipeline"""
    
    # Create versioning system
    versioning = create_versioning_system()
    
    # Start experiment
    experiment_id = versioning.track_experiment(
        name="GAELP_PPO_Training",
        experiment_type=ExperimentType.TRAINING,
        config={
            "algorithm": "PPO",
            "learning_rate": 0.001,
            "batch_size": 32,
            "environment": "EnhancedGAELP"
        },
        description="PPO training on enhanced GAELP environment",
        tags=["ppo", "gaelp", "rl"]
    )
    
    # Simulate training and model saving
    # (In practice, this would be your actual trained model)
    dummy_model = {"weights": "model_data", "architecture": "ppo"}
    
    model_id = versioning.save_model_version(
        model_obj=dummy_model,
        model_name="gaelp_ppo_agent",
        config={
            "algorithm": "PPO",
            "learning_rate": 0.001,
            "batch_size": 32
        },
        metrics={
            "final_roas": 3.5,
            "avg_reward": 1250.0,
            "training_episodes": 1000
        },
        experiment_id=experiment_id,
        tags=["ppo", "baseline"],
        description="First successful PPO training run"
    )
    
    # Update experiment
    versioning.update_experiment(
        experiment_id,
        metrics={"training_completed": True, "final_model": model_id},
        status="completed",
        model_versions=[model_id]
    )
    
    return versioning, experiment_id, model_id


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the versioning system
    versioning, exp_id, model_id = integrate_with_gaelp_training()
    
    print(f"Created experiment: {exp_id}")
    print(f"Saved model: {model_id}")
    
    # Get experiment results
    results = versioning.get_experiment_results(exp_id)
    print(f"Experiment results: {results['summary']}")
    
    # Export report
    report_path = versioning.export_experiment_report(exp_id)
    print(f"Exported report to: {report_path}")