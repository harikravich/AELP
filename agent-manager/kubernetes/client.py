"""
Kubernetes client for managing agent deployments and jobs
"""
import os
import logging
from typing import Dict, List, Optional, Any
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import yaml
import json

logger = logging.getLogger(__name__)


class KubernetesClient:
    """Kubernetes client for agent management"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._init_client()
    
    def _init_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                # Fall back to kubeconfig
                config.load_kube_config()
                logger.info("Loaded kubeconfig")
            except config.ConfigException as e:
                logger.error(f"Could not load Kubernetes config: {e}")
                raise
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.custom_objects = client.CustomObjectsApi()
    
    def create_training_job(
        self, 
        job_name: str,
        agent_name: str,
        docker_image: str,
        command: List[str],
        args: List[str],
        resource_requirements: Dict[str, str],
        environment_vars: Dict[str, str] = None,
        secrets: Dict[str, str] = None,
        node_selector: Dict[str, str] = None
    ) -> bool:
        """Create a Kubernetes job for training"""
        try:
            # Create job specification
            job_spec = self._create_job_spec(
                job_name=job_name,
                agent_name=agent_name,
                docker_image=docker_image,
                command=command,
                args=args,
                resource_requirements=resource_requirements,
                environment_vars=environment_vars or {},
                secrets=secrets or {},
                node_selector=node_selector
            )
            
            # Create the job
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job_spec
            )
            
            logger.info(f"Created training job: {job_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create job {job_name}: {e}")
            return False
    
    def create_agent_deployment(
        self,
        deployment_name: str,
        agent_name: str,
        docker_image: str,
        replicas: int,
        resource_limits: Dict[str, str],
        environment_vars: Dict[str, str] = None,
        ports: List[int] = None,
        node_selector: Dict[str, str] = None
    ) -> bool:
        """Create a Kubernetes deployment for agent"""
        try:
            # Create deployment specification
            deployment_spec = self._create_deployment_spec(
                deployment_name=deployment_name,
                agent_name=agent_name,
                docker_image=docker_image,
                replicas=replicas,
                resource_limits=resource_limits,
                environment_vars=environment_vars or {},
                ports=ports or [],
                node_selector=node_selector
            )
            
            # Create the deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment_spec
            )
            
            logger.info(f"Created deployment: {deployment_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create deployment {deployment_name}: {e}")
            return False
    
    def create_service(
        self,
        service_name: str,
        agent_name: str,
        ports: List[Dict[str, Any]],
        service_type: str = "ClusterIP"
    ) -> bool:
        """Create a Kubernetes service for agent"""
        try:
            service_spec = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=service_name,
                    labels={"agent": agent_name}
                ),
                spec=client.V1ServiceSpec(
                    selector={"agent": agent_name},
                    ports=[
                        client.V1ServicePort(
                            port=port_config["port"],
                            target_port=port_config.get("target_port", port_config["port"]),
                            protocol=port_config.get("protocol", "TCP"),
                            name=port_config.get("name", f"port-{port_config['port']}")
                        )
                        for port_config in ports
                    ],
                    type=service_type
                )
            )
            
            self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service_spec
            )
            
            logger.info(f"Created service: {service_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to create service {service_name}: {e}")
            return False
    
    def get_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            status = {
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "start_time": job.status.start_time,
                "completion_time": job.status.completion_time,
                "conditions": []
            }
            
            if job.status.conditions:
                status["conditions"] = [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_transition_time": condition.last_transition_time
                    }
                    for condition in job.status.conditions
                ]
            
            return status
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get job status for {job_name}: {e}")
            return None
    
    def get_deployment_status(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            return {
                "replicas": deployment.status.replicas or 0,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "unavailable_replicas": deployment.status.unavailable_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
            
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get deployment status for {deployment_name}: {e}")
            return None
    
    def get_pod_logs(self, job_name: str, lines: int = 100) -> Optional[str]:
        """Get logs from job pods"""
        try:
            # Get pods for the job
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )
            
            if not pods.items:
                return None
            
            # Get logs from the first pod
            pod_name = pods.items[0].metadata.name
            logs = self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=lines
            )
            
            return logs
            
        except ApiException as e:
            logger.error(f"Failed to get logs for job {job_name}: {e}")
            return None
    
    def delete_job(self, job_name: str) -> bool:
        """Delete a training job"""
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Background"
            )
            
            logger.info(f"Deleted job: {job_name}")
            return True
            
        except ApiException as e:
            if e.status == 404:
                return True  # Already deleted
            logger.error(f"Failed to delete job {job_name}: {e}")
            return False
    
    def delete_deployment(self, deployment_name: str) -> bool:
        """Delete a deployment"""
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                propagation_policy="Background"
            )
            
            logger.info(f"Deleted deployment: {deployment_name}")
            return True
            
        except ApiException as e:
            if e.status == 404:
                return True  # Already deleted
            logger.error(f"Failed to delete deployment {deployment_name}: {e}")
            return False
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale a deployment"""
        try:
            # Patch the deployment
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=self.namespace,
                body=client.V1Scale(
                    spec=client.V1ScaleSpec(replicas=replicas)
                )
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage for the namespace"""
        try:
            # Get all pods in namespace
            pods = self.v1.list_namespaced_pod(namespace=self.namespace)
            
            total_cpu_requests = 0
            total_memory_requests = 0
            total_cpu_limits = 0
            total_memory_limits = 0
            
            for pod in pods.items:
                if pod.spec.containers:
                    for container in pod.spec.containers:
                        if container.resources:
                            requests = container.resources.requests or {}
                            limits = container.resources.limits or {}
                            
                            # Parse CPU requests/limits
                            if "cpu" in requests:
                                total_cpu_requests += self._parse_cpu(requests["cpu"])
                            if "cpu" in limits:
                                total_cpu_limits += self._parse_cpu(limits["cpu"])
                            
                            # Parse memory requests/limits
                            if "memory" in requests:
                                total_memory_requests += self._parse_memory(requests["memory"])
                            if "memory" in limits:
                                total_memory_limits += self._parse_memory(limits["memory"])
            
            return {
                "cpu_requests": total_cpu_requests,
                "memory_requests": total_memory_requests,
                "cpu_limits": total_cpu_limits,
                "memory_limits": total_memory_limits,
                "pod_count": len(pods.items)
            }
            
        except ApiException as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    def _create_job_spec(
        self,
        job_name: str,
        agent_name: str,
        docker_image: str,
        command: List[str],
        args: List[str],
        resource_requirements: Dict[str, str],
        environment_vars: Dict[str, str],
        secrets: Dict[str, str],
        node_selector: Optional[Dict[str, str]]
    ) -> client.V1Job:
        """Create Kubernetes job specification"""
        
        # Environment variables
        env_vars = [
            client.V1EnvVar(name=k, value=v)
            for k, v in environment_vars.items()
        ]
        
        # Add secrets as environment variables
        for secret_name, secret_key in secrets.items():
            env_vars.append(
                client.V1EnvVar(
                    name=secret_name.upper(),
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name=secret_name,
                            key=secret_key
                        )
                    )
                )
            )
        
        # Container specification
        container = client.V1Container(
            name="agent-trainer",
            image=docker_image,
            command=command,
            args=args,
            env=env_vars,
            resources=client.V1ResourceRequirements(
                requests=resource_requirements,
                limits=resource_requirements
            )
        )
        
        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"agent": agent_name, "job": job_name}
            ),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                node_selector=node_selector
            )
        )
        
        # Job specification
        job_spec = client.V1Job(
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                template=template,
                backoff_limit=3,
                ttl_seconds_after_finished=86400  # 24 hours
            )
        )
        
        return job_spec
    
    def _create_deployment_spec(
        self,
        deployment_name: str,
        agent_name: str,
        docker_image: str,
        replicas: int,
        resource_limits: Dict[str, str],
        environment_vars: Dict[str, str],
        ports: List[int],
        node_selector: Optional[Dict[str, str]]
    ) -> client.V1Deployment:
        """Create Kubernetes deployment specification"""
        
        # Environment variables
        env_vars = [
            client.V1EnvVar(name=k, value=v)
            for k, v in environment_vars.items()
        ]
        
        # Container ports
        container_ports = [
            client.V1ContainerPort(container_port=port)
            for port in ports
        ]
        
        # Container specification
        container = client.V1Container(
            name="agent",
            image=docker_image,
            ports=container_ports,
            env=env_vars,
            resources=client.V1ResourceRequirements(
                limits=resource_limits,
                requests=resource_limits
            )
        )
        
        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"agent": agent_name}
            ),
            spec=client.V1PodSpec(
                containers=[container],
                node_selector=node_selector
            )
        )
        
        # Deployment specification
        deployment_spec = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"agent": agent_name}
                ),
                template=template
            )
        )
        
        return deployment_spec
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to float (in cores)"""
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to float (in GB)"""
        memory_str = memory_str.upper()
        if memory_str.endswith("GI"):
            return float(memory_str[:-2])
        elif memory_str.endswith("G"):
            return float(memory_str[:-1])
        elif memory_str.endswith("MI"):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith("M"):
            return float(memory_str[:-1]) / 1024
        elif memory_str.endswith("KI"):
            return float(memory_str[:-2]) / (1024 * 1024)
        elif memory_str.endswith("K"):
            return float(memory_str[:-1]) / (1024 * 1024)
        return float(memory_str) / (1024 * 1024 * 1024)  # Assume bytes