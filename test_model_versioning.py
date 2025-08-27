#!/usr/bin/env python3
"""
Test suite for the Model Versioning and Experiment Tracking System.
Tests all core functionality including Git integration, W&B tracking, and metadata management.
"""

import unittest
import tempfile
import shutil
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from model_versioning import (
    ModelVersioningSystem,
    ModelStatus,
    ExperimentType,
    ModelMetadata,
    ExperimentMetadata,
    ABTestResult
)


class MockModel:
    """Mock model for testing"""
    def __init__(self, name: str):
        self.name = name
        self.data = f"model_data_for_{name}"


class TestModelVersioningSystem(unittest.TestCase):
    """Test cases for ModelVersioningSystem"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.experiments_dir = os.path.join(self.temp_dir, "experiments")
        
        # Initialize versioning system without git for testing
        self.versioning = ModelVersioningSystem(
            models_dir=self.models_dir,
            experiments_dir=self.experiments_dir,
            git_repo_path=self.temp_dir,
            auto_git_commit=False  # Disable git for testing
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertTrue(Path(self.models_dir).exists())
        self.assertTrue(Path(self.experiments_dir).exists())
        self.assertEqual(len(self.versioning.models_metadata), 0)
        self.assertEqual(len(self.versioning.experiments_metadata), 0)
    
    def test_track_experiment(self):
        """Test experiment tracking"""
        experiment_id = self.versioning.track_experiment(
            name="Test_Experiment",
            experiment_type=ExperimentType.TRAINING,
            config={"param1": "value1", "param2": 42},
            description="Test experiment description",
            tags=["test", "demo"]
        )
        
        self.assertIsNotNone(experiment_id)
        self.assertIn(experiment_id, self.versioning.experiments_metadata)
        
        exp_metadata = self.versioning.experiments_metadata[experiment_id]
        self.assertEqual(exp_metadata.name, "Test_Experiment")
        self.assertEqual(exp_metadata.experiment_type, ExperimentType.TRAINING)
        self.assertEqual(exp_metadata.status, "running")
        self.assertEqual(len(exp_metadata.tags), 2)
    
    def test_save_model_version(self):
        """Test model version saving"""
        # Create experiment first
        experiment_id = self.versioning.track_experiment(
            name="Model_Test_Experiment",
            experiment_type=ExperimentType.TRAINING,
            config={"algorithm": "test"}
        )
        
        # Create and save model
        mock_model = MockModel("test_model")
        model_id = self.versioning.save_model_version(
            model_obj=mock_model,
            model_name="test_agent",
            config={"lr": 0.001, "batch_size": 32},
            metrics={"accuracy": 0.95, "loss": 0.05},
            experiment_id=experiment_id,
            tags=["test", "v1"],
            description="Test model version",
            status=ModelStatus.TRAINING
        )
        
        self.assertIsNotNone(model_id)
        self.assertIn(model_id, self.versioning.models_metadata)
        
        # Check metadata
        model_metadata = self.versioning.models_metadata[model_id]
        self.assertEqual(model_metadata.status, ModelStatus.TRAINING)
        self.assertEqual(model_metadata.experiment_id, experiment_id)
        self.assertEqual(len(model_metadata.tags), 2)
        self.assertEqual(model_metadata.metrics["accuracy"], 0.95)
        
        # Check files were created
        model_path = Path(model_metadata.model_path)
        self.assertTrue(model_path.exists())
        self.assertTrue(model_path.parent.joinpath("config.json").exists())
        self.assertTrue(model_path.parent.joinpath("metrics.json").exists())
    
    def test_update_experiment(self):
        """Test experiment updating"""
        experiment_id = self.versioning.track_experiment(
            name="Update_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        # Update experiment
        self.versioning.update_experiment(
            experiment_id=experiment_id,
            metrics={"final_score": 0.87, "episodes": 1000},
            status="completed"
        )
        
        exp_metadata = self.versioning.experiments_metadata[experiment_id]
        self.assertEqual(exp_metadata.status, "completed")
        self.assertEqual(exp_metadata.metrics["final_score"], 0.87)
        self.assertIsNotNone(exp_metadata.duration_seconds)
    
    def test_compare_versions(self):
        """Test model version comparison"""
        # Create experiment and two models
        experiment_id = self.versioning.track_experiment(
            name="Comparison_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        model1 = MockModel("model1")
        model1_id = self.versioning.save_model_version(
            model_obj=model1,
            model_name="test_agent",
            config={"lr": 0.001},
            metrics={"accuracy": 0.90, "roas": 3.0},
            experiment_id=experiment_id
        )
        
        model2 = MockModel("model2")
        model2_id = self.versioning.save_model_version(
            model_obj=model2,
            model_name="test_agent",
            config={"lr": 0.01},
            metrics={"accuracy": 0.95, "roas": 3.5},
            experiment_id=experiment_id
        )
        
        # Compare models
        comparison = self.versioning.compare_versions(
            model1_id,
            model2_id,
            metrics=["accuracy", "roas"]
        )
        
        self.assertIn("version1", comparison)
        self.assertIn("version2", comparison)
        self.assertIn("differences", comparison)
        self.assertIn("overall_winner", comparison)
        
        # Check accuracy difference
        acc_diff = comparison["differences"]["accuracy"]
        self.assertAlmostEqual(acc_diff["absolute_diff"], 0.05, places=6)
        self.assertEqual(acc_diff["winner"], "version2")
        
        # Check ROAS difference
        roas_diff = comparison["differences"]["roas"]
        self.assertEqual(roas_diff["absolute_diff"], 0.5)
    
    def test_ab_test(self):
        """Test A/B testing functionality"""
        # Create experiment and two models
        experiment_id = self.versioning.track_experiment(
            name="AB_Test_Setup",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        model_a = MockModel("model_a")
        model_a_id = self.versioning.save_model_version(
            model_obj=model_a,
            model_name="agent_a",
            config={"version": "a"},
            metrics={"roas": 3.0},
            experiment_id=experiment_id,
            status=ModelStatus.PRODUCTION
        )
        
        model_b = MockModel("model_b")
        model_b_id = self.versioning.save_model_version(
            model_obj=model_b,
            model_name="agent_b",
            config={"version": "b"},
            metrics={"roas": 3.2},
            experiment_id=experiment_id,
            status=ModelStatus.PRODUCTION
        )
        
        # Run A/B test
        test_id = self.versioning.run_ab_test(
            test_name="A_vs_B_Test",
            model_a_version=model_a_id,
            model_b_version=model_b_id,
            traffic_split={"model_a": 0.6, "model_b": 0.4},
            duration_hours=12.0
        )
        
        self.assertIsNotNone(test_id)
        self.assertIn(test_id, self.versioning.ab_tests)
        
        # Check test result
        test_result = self.versioning.ab_tests[test_id]
        self.assertEqual(test_result.model_a_version, model_a_id)
        self.assertEqual(test_result.model_b_version, model_b_id)
        self.assertEqual(test_result.traffic_split["model_a"], 0.6)
        self.assertIn("model_a", test_result.metrics)
        self.assertIn("model_b", test_result.metrics)
        self.assertIsNotNone(test_result.winner)
    
    def test_rollback(self):
        """Test model rollback functionality"""
        # Create experiment and model
        experiment_id = self.versioning.track_experiment(
            name="Rollback_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        model = MockModel("rollback_model")
        model_id = self.versioning.save_model_version(
            model_obj=model,
            model_name="rollback_agent",
            config={"version": "stable"},
            metrics={"roas": 3.5},
            experiment_id=experiment_id,
            status=ModelStatus.PRODUCTION
        )
        
        # Perform rollback
        success = self.versioning.rollback(
            target_version=model_id,
            reason="Testing rollback functionality"
        )
        
        self.assertTrue(success)
        
        # Check that current model directory exists
        current_dir = Path(self.models_dir) / "current"
        self.assertTrue(current_dir.exists())
    
    def test_model_history(self):
        """Test model history retrieval"""
        experiment_id = self.versioning.track_experiment(
            name="History_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        # Create multiple model versions
        for i in range(3):
            model = MockModel(f"model_{i}")
            self.versioning.save_model_version(
                model_obj=model,
                model_name="history_agent",
                config={"version": i},
                metrics={"score": i * 0.1},
                experiment_id=experiment_id
            )
        
        # Get history
        history = self.versioning.get_model_history("history_agent")
        self.assertEqual(len(history), 3)
        
        # Check ordering (most recent first)
        for i in range(len(history) - 1):
            self.assertGreater(history[i].timestamp, history[i + 1].timestamp)
    
    def test_experiment_results(self):
        """Test experiment results retrieval"""
        experiment_id = self.versioning.track_experiment(
            name="Results_Test",
            experiment_type=ExperimentType.TRAINING,
            config={"test": True}
        )
        
        # Add model to experiment
        model = MockModel("results_model")
        model_id = self.versioning.save_model_version(
            model_obj=model,
            model_name="results_agent",
            config={"version": "1"},
            metrics={"score": 0.8},
            experiment_id=experiment_id
        )
        
        # Update experiment
        self.versioning.update_experiment(
            experiment_id,
            metrics={"final_score": 0.8},
            status="completed",
            model_versions=[model_id]
        )
        
        # Get results
        results = self.versioning.get_experiment_results(experiment_id)
        
        self.assertIn("experiment", results)
        self.assertIn("models", results)
        self.assertIn("summary", results)
        
        self.assertEqual(results["summary"]["total_models"], 1)
        self.assertEqual(len(results["models"]), 1)
        self.assertEqual(results["models"][0]["model_id"], model_id)
    
    def test_metadata_persistence(self):
        """Test metadata saving and loading"""
        # Create some data
        experiment_id = self.versioning.track_experiment(
            name="Persistence_Test",
            experiment_type=ExperimentType.TRAINING,
            config={"persist": True}
        )
        
        model = MockModel("persist_model")
        model_id = self.versioning.save_model_version(
            model_obj=model,
            model_name="persist_agent",
            config={"version": "persist"},
            metrics={"score": 0.9},
            experiment_id=experiment_id
        )
        
        # Save metadata
        self.versioning._save_metadata()
        
        # Create new instance and check data is loaded
        new_versioning = ModelVersioningSystem(
            models_dir=self.models_dir,
            experiments_dir=self.experiments_dir,
            auto_git_commit=False
        )
        
        self.assertIn(experiment_id, new_versioning.experiments_metadata)
        self.assertIn(model_id, new_versioning.models_metadata)
        
        # Check data integrity
        exp_data = new_versioning.experiments_metadata[experiment_id]
        self.assertEqual(exp_data.name, "Persistence_Test")
        
        model_data = new_versioning.models_metadata[model_id]
        self.assertEqual(model_data.metrics["score"], 0.9)
    
    def test_production_models(self):
        """Test production models filtering"""
        experiment_id = self.versioning.track_experiment(
            name="Production_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        # Create models with different statuses
        statuses = [ModelStatus.TRAINING, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED]
        model_ids = []
        
        for i, status in enumerate(statuses):
            model = MockModel(f"prod_model_{i}")
            model_id = self.versioning.save_model_version(
                model_obj=model,
                model_name=f"prod_agent_{i}",
                config={"status_test": True},
                metrics={"score": 0.5},
                experiment_id=experiment_id,
                status=status
            )
            model_ids.append(model_id)
        
        # Get production models
        prod_models = self.versioning.get_production_models()
        self.assertEqual(len(prod_models), 1)
        self.assertEqual(prod_models[0].status, ModelStatus.PRODUCTION)
    
    def test_model_lineage(self):
        """Test model lineage tracking"""
        experiment_id = self.versioning.track_experiment(
            name="Lineage_Test",
            experiment_type=ExperimentType.TRAINING,
            config={}
        )
        
        # Create parent model
        parent_model = MockModel("parent")
        parent_id = self.versioning.save_model_version(
            model_obj=parent_model,
            model_name="lineage_agent",
            config={"generation": 1},
            metrics={"score": 0.7},
            experiment_id=experiment_id
        )
        
        # Create child model
        child_model = MockModel("child")
        child_id = self.versioning.save_model_version(
            model_obj=child_model,
            model_name="lineage_agent",
            config={"generation": 2},
            metrics={"score": 0.8},
            experiment_id=experiment_id,
            parent_version=parent_id
        )
        
        # Get lineage for child
        lineage = self.versioning.get_model_lineage(child_id)
        
        self.assertEqual(lineage["model"]["model_id"], child_id)
        self.assertEqual(len(lineage["parents"]), 1)
        self.assertEqual(lineage["parents"][0]["model_id"], parent_id)
        self.assertEqual(lineage["depth"], 1)
        
        # Get lineage for parent
        parent_lineage = self.versioning.get_model_lineage(parent_id)
        self.assertEqual(len(parent_lineage["children"]), 1)
        self.assertEqual(parent_lineage["children"][0]["model_id"], child_id)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.versioning = ModelVersioningSystem(
            models_dir=os.path.join(self.temp_dir, "models"),
            experiments_dir=os.path.join(self.temp_dir, "experiments"),
            auto_git_commit=False
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_ml_workflow(self):
        """Test complete ML workflow scenario"""
        # 1. Start training experiment
        train_exp_id = self.versioning.track_experiment(
            name="Complete_Workflow_Training",
            experiment_type=ExperimentType.TRAINING,
            config={"algorithm": "PPO", "lr": 0.001}
        )
        
        # 2. Train and save model
        model = MockModel("workflow_model")
        model_id = self.versioning.save_model_version(
            model_obj=model,
            model_name="workflow_agent",
            config={"algorithm": "PPO", "lr": 0.001},
            metrics={"final_roas": 3.2, "training_episodes": 1000},
            experiment_id=train_exp_id,
            status=ModelStatus.TRAINING
        )
        
        # 3. Validate model
        self.versioning.models_metadata[model_id].status = ModelStatus.VALIDATION
        
        # 4. Deploy to production
        self.versioning.models_metadata[model_id].status = ModelStatus.PRODUCTION
        
        # 5. Start hyperparameter sweep
        sweep_exp_id = self.versioning.track_experiment(
            name="Workflow_Hyperparameter_Sweep",
            experiment_type=ExperimentType.HYPERPARAMETER_SWEEP,
            config={"base_model": model_id}
        )
        
        # 6. Create improved model
        improved_model = MockModel("improved_model")
        improved_id = self.versioning.save_model_version(
            model_obj=improved_model,
            model_name="workflow_agent",
            config={"algorithm": "PPO", "lr": 0.003},
            metrics={"final_roas": 3.5, "training_episodes": 800},
            experiment_id=sweep_exp_id,
            parent_version=model_id,
            status=ModelStatus.VALIDATION
        )
        
        # 7. Run A/B test
        test_id = self.versioning.run_ab_test(
            test_name="Baseline_vs_Improved",
            model_a_version=model_id,
            model_b_version=improved_id
        )
        
        # 8. Check results
        test_result = self.versioning.ab_tests[test_id]
        if test_result.winner == "model_b":
            # Deploy improved model
            self.versioning.models_metadata[improved_id].status = ModelStatus.PRODUCTION
            self.versioning.models_metadata[model_id].status = ModelStatus.DEPRECATED
        
        # Validate final state
        prod_models = self.versioning.get_production_models()
        self.assertEqual(len(prod_models), 1)
        
        # Check experiment results
        train_results = self.versioning.get_experiment_results(train_exp_id)
        sweep_results = self.versioning.get_experiment_results(sweep_exp_id)
        
        self.assertEqual(train_results["summary"]["total_models"], 1)
        self.assertGreater(len(self.versioning.models_metadata), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelVersioningSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)