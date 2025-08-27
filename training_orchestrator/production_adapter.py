"""
Production Service Adapter for GAELP Training Orchestrator

This module provides adapters to connect the training orchestrator to real
production services instead of demo/mock services.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import redis
from google.cloud import bigquery, pubsub_v1, storage
from google.auth import default

logger = logging.getLogger(__name__)


@dataclass
class ProductionServiceConfig:
    """Configuration for production services"""
    project_id: str
    region: str
    
    # BigQuery configuration
    bigquery_datasets: Dict[str, str]
    
    # Redis configuration
    redis_cache_host: str
    redis_cache_port: int
    redis_cache_auth: Optional[str] = None
    redis_sessions_host: str
    redis_sessions_port: int
    redis_sessions_auth: Optional[str] = None
    
    # Pub/Sub configuration
    pubsub_topics: Dict[str, str]
    
    # Storage configuration
    storage_buckets: Dict[str, str]


class ProductionServiceAdapter:
    """Adapter for connecting to real GCP services"""
    
    def __init__(self, config: ProductionServiceConfig):
        self.config = config
        self._credentials, self._project = default()
        
        # Initialize clients
        self._bigquery_client = None
        self._pubsub_publisher = None
        self._pubsub_subscriber = None
        self._storage_client = None
        self._redis_cache = None
        self._redis_sessions = None
    
    @property
    def bigquery_client(self) -> bigquery.Client:
        """Get BigQuery client"""
        if self._bigquery_client is None:
            self._bigquery_client = bigquery.Client(
                project=self.config.project_id,
                credentials=self._credentials
            )
        return self._bigquery_client
    
    @property
    def pubsub_publisher(self) -> pubsub_v1.PublisherClient:
        """Get Pub/Sub publisher client"""
        if self._pubsub_publisher is None:
            self._pubsub_publisher = pubsub_v1.PublisherClient(
                credentials=self._credentials
            )
        return self._pubsub_publisher
    
    @property
    def pubsub_subscriber(self) -> pubsub_v1.SubscriberClient:
        """Get Pub/Sub subscriber client"""
        if self._pubsub_subscriber is None:
            self._pubsub_subscriber = pubsub_v1.SubscriberClient(
                credentials=self._credentials
            )
        return self._pubsub_subscriber
    
    @property
    def storage_client(self) -> storage.Client:
        """Get Cloud Storage client"""
        if self._storage_client is None:
            self._storage_client = storage.Client(
                project=self.config.project_id,
                credentials=self._credentials
            )
        return self._storage_client
    
    @property
    def redis_cache(self) -> redis.Redis:
        """Get Redis cache connection"""
        if self._redis_cache is None:
            self._redis_cache = redis.Redis(
                host=self.config.redis_cache_host,
                port=self.config.redis_cache_port,
                password=self.config.redis_cache_auth,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        return self._redis_cache
    
    @property
    def redis_sessions(self) -> redis.Redis:
        """Get Redis sessions connection"""
        if self._redis_sessions is None:
            self._redis_sessions = redis.Redis(
                host=self.config.redis_sessions_host,
                port=self.config.redis_sessions_port,
                password=self.config.redis_sessions_auth,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        return self._redis_sessions
    
    def test_connections(self) -> Dict[str, bool]:
        """Test all service connections"""
        results = {}
        
        # Test BigQuery
        try:
            list(self.bigquery_client.list_datasets(max_results=1))
            results['bigquery'] = True
            logger.info("✅ BigQuery connection successful")
        except Exception as e:
            results['bigquery'] = False
            logger.error(f"❌ BigQuery connection failed: {e}")
        
        # Test Pub/Sub
        try:
            list(self.pubsub_publisher.list_topics(
                request={"project": f"projects/{self.config.project_id}"},
                timeout=5
            ))
            results['pubsub'] = True
            logger.info("✅ Pub/Sub connection successful")
        except Exception as e:
            results['pubsub'] = False
            logger.error(f"❌ Pub/Sub connection failed: {e}")
        
        # Test Cloud Storage
        try:
            list(self.storage_client.list_buckets(max_results=1))
            results['storage'] = True
            logger.info("✅ Cloud Storage connection successful")
        except Exception as e:
            results['storage'] = False
            logger.error(f"❌ Cloud Storage connection failed: {e}")
        
        # Test Redis Cache
        try:
            self.redis_cache.ping()
            results['redis_cache'] = True
            logger.info("✅ Redis cache connection successful")
        except Exception as e:
            results['redis_cache'] = False
            logger.error(f"❌ Redis cache connection failed: {e}")
        
        # Test Redis Sessions
        try:
            self.redis_sessions.ping()
            results['redis_sessions'] = True
            logger.info("✅ Redis sessions connection successful")
        except Exception as e:
            results['redis_sessions'] = False
            logger.error(f"❌ Redis sessions connection failed: {e}")
        
        return results
    
    def store_training_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store training metrics in BigQuery"""
        try:
            dataset_id = self.config.bigquery_datasets.get('training')
            if not dataset_id:
                logger.error("Training dataset not configured")
                return False
            
            table_id = "training_metrics"
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            
            # Insert data
            errors = self.bigquery_client.insert_rows_json(table_ref, [metrics])
            
            if errors:
                logger.error(f"Failed to insert training metrics: {errors}")
                return False
            
            logger.info(f"✅ Training metrics stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store training metrics: {e}")
            return False
    
    def publish_training_event(self, event_data: Dict[str, Any]) -> bool:
        """Publish training event to Pub/Sub"""
        try:
            topic_name = self.config.pubsub_topics.get('training_events')
            if not topic_name:
                logger.error("Training events topic not configured")
                return False
            
            topic_path = self.pubsub_publisher.topic_path(
                self.config.project_id, topic_name
            )
            
            # Publish message
            data = str(event_data).encode('utf-8')
            future = self.pubsub_publisher.publish(topic_path, data)
            future.result()  # Wait for publish to complete
            
            logger.info(f"✅ Training event published successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish training event: {e}")
            return False
    
    def cache_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Cache agent state in Redis"""
        try:
            key = f"agent_state:{agent_id}"
            self.redis_cache.hset(key, mapping=state)
            self.redis_cache.expire(key, 3600)  # 1 hour TTL
            
            logger.info(f"✅ Agent state cached for {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache agent state: {e}")
            return False
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state from Redis cache"""
        try:
            key = f"agent_state:{agent_id}"
            state = self.redis_cache.hgetall(key)
            
            if state:
                logger.info(f"✅ Agent state retrieved for {agent_id}")
                return state
            else:
                logger.info(f"No cached state found for {agent_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get agent state: {e}")
            return None


def create_production_adapter() -> Optional[ProductionServiceAdapter]:
    """Create production service adapter from environment configuration"""
    
    try:
        config = ProductionServiceConfig(
            project_id=os.getenv('PROJECT_ID', 'aura-thrive-platform'),
            region=os.getenv('REGION', 'us-central1'),
            
            bigquery_datasets={
                'campaign': os.getenv('BIGQUERY_DATASET_CAMPAIGN', 'campaign_performance'),
                'training': os.getenv('BIGQUERY_DATASET_TRAINING', 'agent_training'),
                'simulation': os.getenv('BIGQUERY_DATASET_SIMULATION', 'simulation_results'),
                'realtime': os.getenv('BIGQUERY_DATASET_REALTIME', 'realtime_metrics'),
            },
            
            redis_cache_host=os.getenv('REDIS_CACHE_HOST', 'localhost'),
            redis_cache_port=int(os.getenv('REDIS_CACHE_PORT', '6379')),
            redis_cache_auth=os.getenv('REDIS_CACHE_AUTH'),
            
            redis_sessions_host=os.getenv('REDIS_SESSIONS_HOST', 'localhost'),
            redis_sessions_port=int(os.getenv('REDIS_SESSIONS_PORT', '6379')),
            redis_sessions_auth=os.getenv('REDIS_SESSIONS_AUTH'),
            
            pubsub_topics={
                'training_events': os.getenv('PUBSUB_TOPIC_TRAINING_EVENTS', 'gaelp-training-events'),
                'safety_alerts': os.getenv('PUBSUB_TOPIC_SAFETY_ALERTS', 'gaelp-safety-alerts'),
                'campaign_events': os.getenv('PUBSUB_TOPIC_CAMPAIGN_EVENTS', 'gaelp-campaign-events'),
            },
            
            storage_buckets={
                'model_artifacts': os.getenv('GCS_BUCKET_MODEL_ARTIFACTS', ''),
                'training_data': os.getenv('GCS_BUCKET_TRAINING_DATA', ''),
                'campaign_assets': os.getenv('GCS_BUCKET_CAMPAIGN_ASSETS', ''),
                'simulation_results': os.getenv('GCS_BUCKET_SIMULATION_RESULTS', ''),
                'temp_processing': os.getenv('GCS_BUCKET_TEMP_PROCESSING', ''),
            }
        )
        
        adapter = ProductionServiceAdapter(config)
        
        # Test connections
        results = adapter.test_connections()
        
        # Log connection status
        total_services = len(results)
        successful_connections = sum(results.values())
        
        logger.info(f"Service connections: {successful_connections}/{total_services} successful")
        
        if successful_connections == 0:
            logger.warning("No services are accessible - falling back to demo mode")
            return None
        
        return adapter
        
    except Exception as e:
        logger.error(f"Failed to create production adapter: {e}")
        return None


# Global adapter instance
_production_adapter: Optional[ProductionServiceAdapter] = None


def get_production_adapter() -> Optional[ProductionServiceAdapter]:
    """Get the global production adapter instance"""
    global _production_adapter
    if _production_adapter is None:
        _production_adapter = create_production_adapter()
    return _production_adapter