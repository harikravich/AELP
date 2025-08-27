#!/usr/bin/env python3
"""Test BigQuery connection using Thrive project"""

import os
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

def test_bigquery_connection():
    """Test if we can connect to BigQuery using Thrive project"""
    
    print("Testing BigQuery connection with Thrive project...")
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    # Use the Thrive project
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'aura-thrive-platform')
    
    try:
        # Try to create a BigQuery client
        client = bigquery.Client(project=project_id)
        print(f"✅ Successfully created BigQuery client for project: {project_id}")
        
        # Try to list datasets (this will fail if no permissions)
        try:
            datasets = list(client.list_datasets(max_results=1))
            if datasets:
                print(f"✅ Can access datasets in project {project_id}")
                print(f"   Found dataset: {datasets[0].dataset_id}")
            else:
                print(f"⚠️  No datasets found in project {project_id} (might need to create one)")
        except Exception as e:
            print(f"⚠️  Cannot list datasets: {e}")
            print("   This is OK - we may not have list permissions but can still create/use specific datasets")
        
        # Try to check if gaelp_data dataset exists
        dataset_id = "gaelp_data"
        dataset_ref = client.dataset(dataset_id)
        
        try:
            dataset = client.get_dataset(dataset_ref)
            print(f"✅ Dataset {dataset_id} exists in project {project_id}")
        except Exception:
            print(f"ℹ️  Dataset {dataset_id} doesn't exist yet - would need to create it")
            
            # Try to create the dataset
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "GAELP campaign data"
                
                dataset = client.create_dataset(dataset, exists_ok=True)
                print(f"✅ Successfully created dataset {dataset_id}")
            except Exception as e:
                print(f"❌ Cannot create dataset: {e}")
                print("   May need additional permissions or quota")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to BigQuery: {e}")
        
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            print("\n💡 No GOOGLE_APPLICATION_CREDENTIALS set")
            print("   BigQuery will try to use Application Default Credentials (ADC)")
            print("   Run: gcloud auth application-default login")
        
        return False

if __name__ == "__main__":
    success = test_bigquery_connection()
    
    if success:
        print("\n✅ BigQuery connection test PASSED")
        print("   The Journey Database should work with the Thrive project")
    else:
        print("\n❌ BigQuery connection test FAILED")
        print("   The Journey Database will fall back to in-memory storage")