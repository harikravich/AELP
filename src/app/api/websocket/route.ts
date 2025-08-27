import { NextRequest } from 'next/server';
import { WebSocketServer } from 'ws';
import { WebSocketMessage } from '@/types';

// Store WebSocket connections
const clients = new Set<any>();

// Mock data generator for real-time updates
function generateMockUpdate(): WebSocketMessage {
  const updateTypes = ['training_update', 'campaign_update', 'safety_alert', 'system_status'];
  const type = updateTypes[Math.floor(Math.random() * updateTypes.length)] as WebSocketMessage['type'];
  
  let data;
  switch (type) {
    case 'training_update':
      data = {
        agentId: `agent-${Math.floor(Math.random() * 100)}`,
        episode: Math.floor(Math.random() * 1000),
        reward: Math.random() * 100,
        message: 'Training progress updated',
      };
      break;
    case 'campaign_update':
      data = {
        campaignId: `campaign-${Math.floor(Math.random() * 50)}`,
        spend: Math.random() * 1000,
        revenue: Math.random() * 2000,
        message: 'Campaign metrics updated',
      };
      break;
    case 'safety_alert':
      data = {
        agentId: `agent-${Math.floor(Math.random() * 100)}`,
        severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
        message: 'Safety event detected',
      };
      break;
    case 'system_status':
      data = {
        component: ['training', 'campaigns', 'safety', 'api'][Math.floor(Math.random() * 4)],
        status: ['healthy', 'degraded', 'error'][Math.floor(Math.random() * 3)],
        message: 'System status updated',
      };
      break;
  }

  return {
    type,
    data,
    timestamp: new Date(),
  };
}

// Simulate real-time updates
setInterval(() => {
  if (clients.size > 0) {
    const update = generateMockUpdate();
    const message = JSON.stringify(update);
    
    clients.forEach(client => {
      if (client.readyState === 1) { // WebSocket.OPEN
        client.send(message);
      }
    });
  }
}, 5000); // Send update every 5 seconds

export async function GET(request: NextRequest) {
  // Check if the request is a WebSocket upgrade
  const upgradeHeader = request.headers.get('upgrade');
  
  if (upgradeHeader?.toLowerCase() === 'websocket') {
    return new Response('WebSocket upgrade required', { status: 426 });
  }

  return new Response('WebSocket endpoint', { status: 200 });
}

// Note: Next.js doesn't directly support WebSocket upgrades in API routes
// In a production environment, you would typically handle WebSockets separately
// using a dedicated WebSocket server or a service like Socket.IO

// For development, you might want to use a separate WebSocket server
// or implement Server-Sent Events (SSE) as an alternative

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, data } = body;

    // Broadcast message to all connected clients
    const message: WebSocketMessage = {
      type,
      data,
      timestamp: new Date(),
    };

    const messageString = JSON.stringify(message);
    
    clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(messageString);
      }
    });

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error broadcasting WebSocket message:', error);
    return new Response(JSON.stringify({ error: 'Failed to broadcast message' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}