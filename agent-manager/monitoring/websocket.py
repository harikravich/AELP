"""
WebSocket manager for real-time monitoring
"""
import json
import logging
from typing import Dict, List, Any, Set
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time monitoring"""
    
    def __init__(self):
        # agent_id -> set of websockets
        self.agent_connections: Dict[int, Set[WebSocket]] = {}
        # All active connections
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, agent_id: int):
        """Connect a WebSocket for agent monitoring"""
        try:
            await websocket.accept()
            
            # Add to active connections
            self.active_connections.add(websocket)
            
            # Add to agent-specific connections
            if agent_id not in self.agent_connections:
                self.agent_connections[agent_id] = set()
            self.agent_connections[agent_id].add(websocket)
            
            logger.info(f"WebSocket connected for agent {agent_id}")
            
            # Send initial connection message
            await websocket.send_text(json.dumps({
                "type": "connection",
                "message": f"Connected to agent {agent_id} monitoring",
                "agent_id": agent_id
            }))
            
            # Keep connection alive and handle messages
            await self._handle_connection(websocket, agent_id)
            
        except WebSocketDisconnect:
            await self.disconnect(websocket, agent_id)
        except Exception as e:
            logger.error(f"WebSocket connection error for agent {agent_id}: {e}")
            await self.disconnect(websocket, agent_id)
    
    async def disconnect(self, websocket: WebSocket, agent_id: int):
        """Disconnect a WebSocket"""
        try:
            # Remove from active connections
            self.active_connections.discard(websocket)
            
            # Remove from agent-specific connections
            if agent_id in self.agent_connections:
                self.agent_connections[agent_id].discard(websocket)
                
                # Clean up empty agent connection sets
                if not self.agent_connections[agent_id]:
                    del self.agent_connections[agent_id]
            
            logger.info(f"WebSocket disconnected for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket for agent {agent_id}: {e}")
    
    async def _handle_connection(self, websocket: WebSocket, agent_id: int):
        """Handle WebSocket connection lifecycle"""
        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }))
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific metrics
                    await self._handle_subscription(websocket, agent_id, message)
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error handling WebSocket connection for agent {agent_id}: {e}")
    
    async def _handle_subscription(self, websocket: WebSocket, agent_id: int, message: Dict[str, Any]):
        """Handle metric subscription requests"""
        metrics_type = message.get("metrics_type", "all")
        
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "agent_id": agent_id,
            "metrics_type": metrics_type,
            "message": f"Subscribed to {metrics_type} metrics for agent {agent_id}"
        }))
    
    async def broadcast_agent_metrics(self, agent_id: int, metrics: Dict[str, Any]):
        """Broadcast metrics to all connections monitoring this agent"""
        if agent_id not in self.agent_connections:
            return
        
        message = {
            "type": "metrics",
            "agent_id": agent_id,
            "data": metrics,
            "timestamp": metrics.get("timestamp")
        }
        
        # Send to all connections for this agent
        disconnected = set()
        for websocket in self.agent_connections[agent_id].copy():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send metrics to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket, agent_id)
    
    async def broadcast_job_status(self, agent_id: int, job_id: int, status: str, details: Dict[str, Any] = None):
        """Broadcast job status updates"""
        if agent_id not in self.agent_connections:
            return
        
        message = {
            "type": "job_status",
            "agent_id": agent_id,
            "job_id": job_id,
            "status": status,
            "details": details or {},
            "timestamp": details.get("timestamp") if details else None
        }
        
        # Send to all connections for this agent
        disconnected = set()
        for websocket in self.agent_connections[agent_id].copy():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send job status to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket, agent_id)
    
    async def broadcast_system_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None):
        """Broadcast system-wide alerts"""
        alert_message = {
            "type": "system_alert",
            "alert_type": alert_type,
            "message": message,
            "details": details or {},
            "timestamp": details.get("timestamp") if details else None
        }
        
        # Send to all active connections
        disconnected = set()
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(json.dumps(alert_message))
            except Exception as e:
                logger.warning(f"Failed to send system alert to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.active_connections.discard(websocket)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_agent_connection_count(self, agent_id: int) -> int:
        """Get number of connections for a specific agent"""
        return len(self.agent_connections.get(agent_id, set()))
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "total_connections": len(self.active_connections),
            "agents_monitored": len(self.agent_connections),
            "connections_per_agent": {
                agent_id: len(connections) 
                for agent_id, connections in self.agent_connections.items()
            }
        }