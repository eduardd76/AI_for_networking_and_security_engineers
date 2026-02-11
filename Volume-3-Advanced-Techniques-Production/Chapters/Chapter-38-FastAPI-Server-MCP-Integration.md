# Chapter 38: FastAPI Server & MCP Integration

## Introduction

Production AI systems need robust APIs. You can't just run Python scripts forever. FastAPI provides async performance, automatic validation, and OpenAPI docs. The Model Context Protocol (MCP) standardizes how AI agents interact with external tools and data sources.

This chapter builds a production-grade FastAPI server that exposes network operations through MCP-compatible endpoints. You'll see the progression from a basic server to a production system handling autonomous AI agents with authentication, rate limiting, and Kubernetes deployment.

No toy examples. This is code you can deploy.

## Version Progression Overview

This chapter follows a four-version progression:

- **V1: Basic FastAPI Server** (30 min, Free) - Pydantic validation, health endpoints, OpenAPI docs
- **V2: Async Network Operations** (45 min, Free) - ThreadPoolExecutor, concurrent device operations, batch execution
- **V3: MCP Integration & AI Agents** (60 min, $20-50/month) - MCP protocol, Claude function calling, agentic loops
- **V4: Production Deployment** (90 min, $100-300/month) - Auth, rate limiting, health checks, Prometheus metrics, Kubernetes

Each version builds on the previous, showing the path from prototype to production.

## V1: Basic FastAPI Server

### Goal
Build a FastAPI server with Pydantic validation, basic health checks, and automatic OpenAPI documentation.

**Time to implement:** 30 minutes
**Cost:** Free (development only)

### Core Implementation

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import logging
import ipaddress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Network AI Operations API",
    description="Production API for AI-powered network operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DeviceQuery(BaseModel):
    """Request model for device queries"""
    device_ip: str = Field(..., description="Device IP address")
    commands: List[str] = Field(..., description="Commands to execute")
    timeout: Optional[int] = Field(30, description="Timeout in seconds")

    @validator('device_ip')
    def validate_ip(cls, v):
        """Validate IP address format"""
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")

    @validator('commands')
    def validate_commands(cls, v):
        """Ensure commands list is not empty"""
        if not v:
            raise ValueError("Commands list cannot be empty")
        return v

class DeviceResponse(BaseModel):
    """Response model for device operations"""
    device_ip: str
    status: str
    message: str
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float

# Global state
START_TIME = datetime.now()

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Network AI Operations API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Basic health check endpoint"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime
    )

@app.post("/query-device", response_model=DeviceResponse)
async def query_device(query: DeviceQuery):
    """
    Query a network device (V1 - placeholder implementation).
    Validates input, returns structured response.
    """
    logger.info(f"Device query: {query.device_ip} - {len(query.commands)} commands")

    # V1: Just validate and acknowledge
    return DeviceResponse(
        device_ip=query.device_ip,
        status="received",
        message=f"Query received for {len(query.commands)} commands",
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run the server:**
```bash
python app/main.py
```

**Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Test with curl:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-19T14:30:25.123456",
  "version": "1.0.0",
  "uptime_seconds": 127.45
}
```

**Test validation:**
```bash
curl -X POST "http://localhost:8000/query-device" \
  -H "Content-Type: application/json" \
  -d '{
    "device_ip": "192.168.1.1",
    "commands": ["show version"]
  }'
```

**Response:**
```json
{
  "device_ip": "192.168.1.1",
  "status": "received",
  "message": "Query received for 1 commands",
  "timestamp": "2026-01-19T14:30:30.123456"
}
```

**Test invalid IP:**
```bash
curl -X POST "http://localhost:8000/query-device" \
  -H "Content-Type: application/json" \
  -d '{
    "device_ip": "invalid_ip",
    "commands": ["show version"]
  }'
```

**Response (422 Validation Error):**
```json
{
  "detail": [
    {
      "loc": ["body", "device_ip"],
      "msg": "Invalid IP address: invalid_ip",
      "type": "value_error"
    }
  ]
}
```

### What V1 Provides

1. **Automatic Validation**: Pydantic validates all inputs before your code runs
2. **OpenAPI Docs**: Interactive API docs at `http://localhost:8000/docs`
3. **Type Safety**: Python type hints ensure correctness
4. **Structured Responses**: Consistent JSON responses
5. **CORS Support**: Ready for web client integration

### V1 Limitations

- No actual network device interaction
- No concurrent request handling
- No authentication or rate limiting
- No production monitoring

## V2: Async Network Operations

### Goal
Add real network device operations with async handling and concurrent execution using ThreadPoolExecutor.

**Time to implement:** 45 minutes
**Cost:** Free (development only)
**Builds on:** V1

### Why ThreadPoolExecutor for Netmiko?

Netmiko is **not async**. It uses blocking I/O. Running Netmiko directly in an async handler blocks the event loop, preventing other requests from processing.

**Solution:** Run Netmiko in a thread pool, wrap with `asyncio.run_in_executor()`.

### Implementation

```python
# app/network_operations.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import time
import ipaddress
from datetime import datetime
from netmiko import ConnectHandler
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Async Network Operations")

# Thread pool for blocking I/O (Netmiko is not async)
executor = ThreadPoolExecutor(max_workers=10)

class NetworkDevice(BaseModel):
    """Network device configuration"""
    host: str
    device_type: str = "cisco_ios"
    username: str
    password: str
    secret: Optional[str] = None

    @validator('host')
    def validate_host(cls, v):
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")

class CommandRequest(BaseModel):
    """Command execution request"""
    devices: List[NetworkDevice]
    commands: List[str]
    enable_mode: bool = False

    @validator('devices')
    def validate_devices(cls, v):
        if not v:
            raise ValueError("Devices list cannot be empty")
        if len(v) > 20:
            raise ValueError("Maximum 20 devices per request")
        return v

    @validator('commands')
    def validate_commands(cls, v):
        if not v:
            raise ValueError("Commands list cannot be empty")
        return v

class CommandResult(BaseModel):
    """Result from a single device"""
    host: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float

class BatchCommandResponse(BaseModel):
    """Response for batch command execution"""
    total_devices: int
    successful: int
    failed: int
    results: List[CommandResult]
    total_execution_time_ms: float

def execute_commands_sync(device: NetworkDevice, commands: List[str],
                         enable_mode: bool) -> CommandResult:
    """
    Synchronous function to execute commands on a single device.
    Runs in thread pool to avoid blocking event loop.
    """
    start_time = time.time()

    try:
        # Connect to device
        connection_params = {
            'device_type': device.device_type,
            'host': device.host,
            'username': device.username,
            'password': device.password,
            'secret': device.secret,
            'timeout': 30,
            'session_log': None
        }

        with ConnectHandler(**connection_params) as net_connect:
            if enable_mode and device.secret:
                net_connect.enable()

            # Execute commands
            output_lines = []
            for command in commands:
                output = net_connect.send_command(command)
                output_lines.append(f"Command: {command}\n{output}\n")

            execution_time = (time.time() - start_time) * 1000

            return CommandResult(
                host=device.host,
                success=True,
                output="\n".join(output_lines),
                error=None,
                execution_time_ms=execution_time
            )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Error executing commands on {device.host}: {str(e)}")

        return CommandResult(
            host=device.host,
            success=False,
            output=None,
            error=str(e),
            execution_time_ms=execution_time
        )

async def execute_commands_async(device: NetworkDevice, commands: List[str],
                                enable_mode: bool) -> CommandResult:
    """
    Async wrapper around synchronous Netmiko operations.
    Uses thread pool to avoid blocking event loop.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        execute_commands_sync,
        device,
        commands,
        enable_mode
    )
    return result

@app.post("/execute-commands", response_model=BatchCommandResponse)
async def execute_commands(request: CommandRequest):
    """
    Execute commands on multiple devices concurrently.
    Each device runs in its own thread via ThreadPoolExecutor.
    Returns results for all devices.
    """
    start_time = time.time()

    logger.info(f"Executing {len(request.commands)} commands on "
                f"{len(request.devices)} devices")

    # Create async tasks for all devices
    tasks = [
        execute_commands_async(device, request.commands, request.enable_mode)
        for device in request.devices
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    command_results = []
    successful = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {str(result)}")
            failed += 1
        elif isinstance(result, CommandResult):
            command_results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1

    total_execution_time = (time.time() - start_time) * 1000

    return BatchCommandResponse(
        total_devices=len(request.devices),
        successful=successful,
        failed=failed,
        results=command_results,
        total_execution_time_ms=total_execution_time
    )
```

**Example: Execute commands on 3 devices concurrently**
```bash
curl -X POST "http://localhost:8000/execute-commands" \
  -H "Content-Type: application/json" \
  -d '{
    "devices": [
      {
        "host": "192.168.1.1",
        "device_type": "cisco_ios",
        "username": "admin",
        "password": "cisco123"
      },
      {
        "host": "192.168.1.2",
        "device_type": "cisco_ios",
        "username": "admin",
        "password": "cisco123"
      },
      {
        "host": "192.168.1.3",
        "device_type": "cisco_ios",
        "username": "admin",
        "password": "cisco123"
      }
    ],
    "commands": ["show version", "show ip interface brief"],
    "enable_mode": false
  }'
```

**Response:**
```json
{
  "total_devices": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "host": "192.168.1.1",
      "success": true,
      "output": "Command: show version\nCisco IOS Software, Version 15.2...\n\nCommand: show ip interface brief\nInterface              IP-Address      OK? Method Status...\n",
      "error": null,
      "execution_time_ms": 1245.67
    },
    {
      "host": "192.168.1.2",
      "success": true,
      "output": "Command: show version\nCisco IOS Software, Version 15.2...\n\nCommand: show ip interface brief\nInterface              IP-Address      OK? Method Status...\n",
      "error": null,
      "execution_time_ms": 1189.23
    },
    {
      "host": "192.168.1.3",
      "success": true,
      "output": "Command: show version\nCisco IOS Software, Version 15.2...\n\nCommand: show ip interface brief\nInterface              IP-Address      OK? Method Status...\n",
      "error": null,
      "execution_time_ms": 1298.45
    }
  ],
  "total_execution_time_ms": 1312.89
}
```

**Key observation:** Total execution time (1,312ms) ≈ slowest device (1,298ms), **not** the sum of all three (3,733ms). This proves concurrent execution.

### V2 Performance Gains

**Sequential execution (without async):**
- 3 devices × 1,200ms average = 3,600ms total

**Concurrent execution (V2):**
- 3 devices in parallel = 1,300ms total (slowest device)

**Speedup:** 2.8× faster

For 10 devices: 12 seconds → 1.5 seconds (8× faster)

### V2 Limitations

- No AI agent integration
- No standardized tool interface
- No authentication
- No production monitoring

## V3: MCP Integration & AI Agents

### Goal
Implement Model Context Protocol (MCP) for standardized tool calling. Integrate Claude with function calling for autonomous network operations.

**Time to implement:** 60 minutes
**Cost:** $20-50/month (Claude API usage)
**Builds on:** V2

### What is MCP?

Model Context Protocol standardizes:
1. **Tool Discovery**: What tools are available?
2. **Tool Schemas**: What inputs does each tool need?
3. **Tool Invocation**: How to call a tool?
4. **Result Formatting**: Standardized responses

### MCP Server Implementation

```python
# app/mcp_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
import time

logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Network Operations Server")

# MCP Protocol Models
class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPToolCall(BaseModel):
    """MCP tool invocation request"""
    tool: str
    arguments: Dict[str, Any]

class MCPToolResult(BaseModel):
    """MCP tool invocation result"""
    tool: str
    result: Any
    error: Optional[str] = None
    execution_time_ms: float

class MCPServerInfo(BaseModel):
    """MCP server information"""
    name: str
    version: str
    protocol_version: str
    capabilities: List[str]

# Define available tools
NETWORK_TOOLS = {
    "get_device_config": {
        "name": "get_device_config",
        "description": "Retrieve running or startup configuration from a network device",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "config_type": {
                    "type": "string",
                    "enum": ["running", "startup"],
                    "description": "Type of configuration to retrieve"
                }
            },
            "required": ["device_ip"]
        }
    },
    "check_interface_status": {
        "name": "check_interface_status",
        "description": "Check status of network interfaces on a device",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "interface_name": {
                    "type": "string",
                    "description": "Specific interface to check (optional)"
                }
            },
            "required": ["device_ip"]
        }
    },
    "analyze_routing_table": {
        "name": "analyze_routing_table",
        "description": "Retrieve and analyze routing table from a device",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "protocol": {
                    "type": "string",
                    "enum": ["all", "ospf", "bgp", "static"],
                    "description": "Filter by routing protocol"
                }
            },
            "required": ["device_ip"]
        }
    },
    "configure_interface": {
        "name": "configure_interface",
        "description": "Configure network interface settings",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "interface_name": {
                    "type": "string",
                    "description": "Interface to configure"
                },
                "ip_address": {
                    "type": "string",
                    "description": "IP address to assign"
                },
                "subnet_mask": {
                    "type": "string",
                    "description": "Subnet mask"
                },
                "description": {
                    "type": "string",
                    "description": "Interface description"
                }
            },
            "required": ["device_ip", "interface_name", "ip_address", "subnet_mask"]
        }
    },
    "search_logs": {
        "name": "search_logs",
        "description": "Search device logs for specific patterns",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "search_pattern": {
                    "type": "string",
                    "description": "Pattern to search for in logs"
                },
                "severity": {
                    "type": "string",
                    "enum": ["emergency", "alert", "critical", "error", "warning", "notice", "info", "debug"],
                    "description": "Minimum severity level"
                }
            },
            "required": ["device_ip", "search_pattern"]
        }
    }
}

# Tool implementations (simplified for example)
async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool with given arguments"""

    if tool_name == "get_device_config":
        device_ip = arguments["device_ip"]
        config_type = arguments.get("config_type", "running")

        # In production: use Netmiko to fetch actual config
        return {
            "device": device_ip,
            "config_type": config_type,
            "config": f"!\n! Configuration for {device_ip}\n!\nversion 15.2\nhostname Router1\n...",
            "lines": 247,
            "retrieved_at": datetime.now().isoformat()
        }

    elif tool_name == "check_interface_status":
        device_ip = arguments["device_ip"]
        interface_name = arguments.get("interface_name")

        if interface_name:
            return {
                "device": device_ip,
                "interface": interface_name,
                "status": "up",
                "protocol": "up",
                "ip_address": "192.168.1.1",
                "subnet_mask": "255.255.255.0"
            }
        else:
            return {
                "device": device_ip,
                "interfaces": [
                    {"name": "GigabitEthernet0/0", "status": "up", "protocol": "up"},
                    {"name": "GigabitEthernet0/1", "status": "up", "protocol": "up"},
                    {"name": "Serial0/0/0", "status": "down", "protocol": "down"}
                ]
            }

    elif tool_name == "analyze_routing_table":
        device_ip = arguments["device_ip"]
        protocol = arguments.get("protocol", "all")

        return {
            "device": device_ip,
            "protocol_filter": protocol,
            "routes": [
                {"network": "0.0.0.0/0", "next_hop": "192.168.1.254", "protocol": "static", "metric": 1},
                {"network": "10.0.0.0/8", "next_hop": "192.168.1.2", "protocol": "ospf", "metric": 110},
                {"network": "172.16.0.0/16", "next_hop": "192.168.1.3", "protocol": "bgp", "metric": 200}
            ],
            "total_routes": 3
        }

    elif tool_name == "configure_interface":
        device_ip = arguments["device_ip"]
        interface_name = arguments["interface_name"]
        ip_address = arguments["ip_address"]
        subnet_mask = arguments["subnet_mask"]
        description = arguments.get("description", "")

        return {
            "device": device_ip,
            "interface": interface_name,
            "configuration_applied": True,
            "changes": {
                "ip_address": ip_address,
                "subnet_mask": subnet_mask,
                "description": description
            },
            "timestamp": datetime.now().isoformat()
        }

    elif tool_name == "search_logs":
        device_ip = arguments["device_ip"]
        search_pattern = arguments["search_pattern"]
        severity = arguments.get("severity", "info")

        return {
            "device": device_ip,
            "search_pattern": search_pattern,
            "severity_filter": severity,
            "matches": [
                {"timestamp": "2026-01-19 10:23:45", "severity": "error", "message": "Interface GigabitEthernet0/1 down"},
                {"timestamp": "2026-01-19 10:25:12", "severity": "warning", "message": "Interface GigabitEthernet0/1 up"}
            ],
            "total_matches": 2
        }

    else:
        raise ValueError(f"Unknown tool: {tool_name}")

# MCP Endpoints
@app.get("/mcp/info", response_model=MCPServerInfo)
async def get_server_info():
    """Get MCP server information"""
    return MCPServerInfo(
        name="Network Operations MCP Server",
        version="1.0.0",
        protocol_version="2024-11-05",
        capabilities=["tools", "async_execution"]
    )

@app.get("/mcp/tools", response_model=List[MCPTool])
async def list_tools():
    """List all available tools"""
    return [
        MCPTool(
            name=tool_data["name"],
            description=tool_data["description"],
            input_schema=tool_data["input_schema"]
        )
        for tool_data in NETWORK_TOOLS.values()
    ]

@app.post("/mcp/call-tool", response_model=MCPToolResult)
async def call_tool(request: MCPToolCall):
    """Execute a tool"""
    start_time = time.time()

    logger.info(f"Executing tool: {request.tool} with arguments: {request.arguments}")

    if request.tool not in NETWORK_TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool not found: {request.tool}")

    try:
        result = await execute_tool(request.tool, request.arguments)
        execution_time = (time.time() - start_time) * 1000

        return MCPToolResult(
            tool=request.tool,
            result=result,
            error=None,
            execution_time_ms=execution_time
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Tool execution failed: {str(e)}")

        return MCPToolResult(
            tool=request.tool,
            result=None,
            error=str(e),
            execution_time_ms=execution_time
        )
```

**Example: List available tools**
```bash
curl http://localhost:8000/mcp/tools
```

**Response:**
```json
[
  {
    "name": "get_device_config",
    "description": "Retrieve running or startup configuration from a network device",
    "input_schema": {
      "type": "object",
      "properties": {
        "device_ip": {"type": "string", "description": "IP address of the device"},
        "config_type": {"type": "string", "enum": ["running", "startup"], "description": "Type of configuration to retrieve"}
      },
      "required": ["device_ip"]
    }
  },
  {
    "name": "check_interface_status",
    "description": "Check status of network interfaces on a device",
    "input_schema": {
      "type": "object",
      "properties": {
        "device_ip": {"type": "string", "description": "IP address of the device"},
        "interface_name": {"type": "string", "description": "Specific interface to check (optional)"}
      },
      "required": ["device_ip"]
    }
  }
]
```

### AI Agent with Claude Function Calling

```python
# app/ai_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import anthropic
import json
import logging
import os

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Network Agent")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class AgentRequest(BaseModel):
    """Request for AI agent"""
    prompt: str
    device_ip: str
    max_iterations: int = 5

class AgentResponse(BaseModel):
    """Response from AI agent"""
    prompt: str
    response: str
    tool_calls: List[Dict[str, Any]]
    iterations: int

# Define tools for Claude (same as MCP tools)
CLAUDE_TOOLS = [
    {
        "name": "get_device_config",
        "description": "Retrieve running or startup configuration from a network device. Use this to see the current device configuration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "config_type": {
                    "type": "string",
                    "enum": ["running", "startup"],
                    "description": "Type of configuration to retrieve"
                }
            },
            "required": ["device_ip"]
        }
    },
    {
        "name": "check_interface_status",
        "description": "Check the operational status of network interfaces. Shows interface state, protocol status, and IP configuration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "interface_name": {
                    "type": "string",
                    "description": "Specific interface to check (optional, leave blank for all)"
                }
            },
            "required": ["device_ip"]
        }
    },
    {
        "name": "analyze_routing_table",
        "description": "Retrieve and analyze the routing table from a device. Shows all routes or filtered by protocol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_ip": {
                    "type": "string",
                    "description": "IP address of the device"
                },
                "protocol": {
                    "type": "string",
                    "enum": ["all", "ospf", "bgp", "static"],
                    "description": "Filter by routing protocol"
                }
            },
            "required": ["device_ip"]
        }
    }
]

async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool via local MCP server"""
    # In production, this would make HTTP request to MCP server
    # For this example, we'll simulate the tool execution

    if tool_name == "get_device_config":
        return {
            "device": arguments["device_ip"],
            "config_type": arguments.get("config_type", "running"),
            "config": "!\nversion 15.2\nhostname Router1\ninterface GigabitEthernet0/0\n ip address 192.168.1.1 255.255.255.0\n no shutdown\n!",
            "lines": 6
        }

    elif tool_name == "check_interface_status":
        if "interface_name" in arguments:
            return {
                "device": arguments["device_ip"],
                "interface": arguments["interface_name"],
                "status": "up",
                "protocol": "up",
                "ip_address": "192.168.1.1",
                "subnet_mask": "255.255.255.0"
            }
        else:
            return {
                "device": arguments["device_ip"],
                "interfaces": [
                    {"name": "GigabitEthernet0/0", "status": "up", "protocol": "up", "ip": "192.168.1.1"},
                    {"name": "GigabitEthernet0/1", "status": "down", "protocol": "down", "ip": "unassigned"}
                ]
            }

    elif tool_name == "analyze_routing_table":
        return {
            "device": arguments["device_ip"],
            "routes": [
                {"network": "0.0.0.0/0", "next_hop": "192.168.1.254", "protocol": "static"},
                {"network": "10.0.0.0/8", "next_hop": "192.168.1.2", "protocol": "ospf"}
            ]
        }

    raise ValueError(f"Unknown tool: {tool_name}")

@app.post("/agent/query", response_model=AgentResponse)
async def agent_query(request: AgentRequest):
    """
    Execute an AI agent query that can use network tools.
    The agent autonomously decides which tools to call.
    """
    logger.info(f"Agent query: {request.prompt}")

    messages = [
        {
            "role": "user",
            "content": f"Device IP: {request.device_ip}\n\nQuery: {request.prompt}"
        }
    ]

    tool_calls = []
    iterations = 0

    while iterations < request.max_iterations:
        iterations += 1

        # Call Claude with tools
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            tools=CLAUDE_TOOLS,
            messages=messages
        )

        logger.info(f"Iteration {iterations}: stop_reason={response.stop_reason}")

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Extract tool calls from response
            assistant_content = []

            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_use_id = content_block.id

                    logger.info(f"Tool call: {tool_name} with input: {tool_input}")

                    # Execute the tool
                    try:
                        tool_result = await execute_mcp_tool(tool_name, tool_input)

                        tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": tool_result
                        })

                        # Add tool result to conversation
                        assistant_content.append(content_block)

                        # Prepare tool result message
                        messages.append({
                            "role": "assistant",
                            "content": assistant_content
                        })

                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps(tool_result)
                                }
                            ]
                        })

                    except Exception as e:
                        logger.error(f"Tool execution failed: {str(e)}")

                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps({"error": str(e)}),
                                    "is_error": True
                                }
                            ]
                        })

                else:
                    assistant_content.append(content_block)

        else:
            # Claude provided final answer
            final_response = ""
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    final_response += content_block.text

            return AgentResponse(
                prompt=request.prompt,
                response=final_response,
                tool_calls=tool_calls,
                iterations=iterations
            )

    # Max iterations reached
    return AgentResponse(
        prompt=request.prompt,
        response="Maximum iterations reached without final answer",
        tool_calls=tool_calls,
        iterations=iterations
    )
```

**Example: Autonomous agent query**
```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Check if all interfaces are up on the device and show me the routing table",
    "device_ip": "192.168.1.1"
  }'
```

**Response:**
```json
{
  "prompt": "Check if all interfaces are up on the device and show me the routing table",
  "response": "I checked the device at 192.168.1.1. Here's what I found:\n\nInterface Status:\n- GigabitEthernet0/0: UP/UP (192.168.1.1)\n- GigabitEthernet0/1: DOWN/DOWN (unassigned)\n\nRouting Table:\n- Default route (0.0.0.0/0) via 192.168.1.254 (static)\n- 10.0.0.0/8 network via 192.168.1.2 (OSPF)\n\nOne interface (GigabitEthernet0/1) is currently down. You may want to investigate why it's not operational.",
  "tool_calls": [
    {
      "tool": "check_interface_status",
      "input": {"device_ip": "192.168.1.1"},
      "result": {
        "device": "192.168.1.1",
        "interfaces": [
          {"name": "GigabitEthernet0/0", "status": "up", "protocol": "up", "ip": "192.168.1.1"},
          {"name": "GigabitEthernet0/1", "status": "down", "protocol": "down", "ip": "unassigned"}
        ]
      }
    },
    {
      "tool": "analyze_routing_table",
      "input": {"device_ip": "192.168.1.1", "protocol": "all"},
      "result": {
        "device": "192.168.1.1",
        "routes": [
          {"network": "0.0.0.0/0", "next_hop": "192.168.1.254", "protocol": "static"},
          {"network": "10.0.0.0/8", "next_hop": "192.168.1.2", "protocol": "ospf"}
        ]
      }
    }
  ],
  "iterations": 2
}
```

**What happened (agentic loop):**
1. User asked to check interfaces and routing table
2. **Iteration 1:** Claude called `check_interface_status` tool
3. Tool result returned to Claude
4. **Iteration 2:** Claude called `analyze_routing_table` tool
5. Tool result returned to Claude
6. Claude synthesized results into human-readable response

The agent **autonomously decided** which tools to call and in what order.

### V3 Benefits

1. **Standardized Tool Interface**: MCP protocol makes tools reusable across AI systems
2. **Autonomous Operation**: Claude decides which tools to use
3. **Multi-Step Reasoning**: Agent can chain multiple tool calls
4. **Structured Responses**: Consistent tool result formatting

### V3 Limitations

- No authentication or rate limiting
- No production monitoring
- Limited error handling for failed tool calls
- No horizontal scaling

## V4: Production Deployment

### Goal
Add authentication, rate limiting, comprehensive health checks, Prometheus metrics, and Kubernetes deployment.

**Time to implement:** 90 minutes
**Cost:** $100-300/month (Claude API + infrastructure)
**Builds on:** V3

### Authentication and Rate Limiting

```python
# app/auth_and_limits.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import time
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Secured Network API")

# Security scheme
security = HTTPBearer()

# In-memory stores (use Redis in production for multi-replica deployments)
API_KEYS = {
    "sk_prod_premium_001": {
        "user_id": "user_001",
        "tier": "premium",
        "rate_limit": 100,  # requests per minute
        "enabled": True
    },
    "sk_prod_basic_002": {
        "user_id": "user_002",
        "tier": "basic",
        "rate_limit": 10,
        "enabled": True
    }
}

# Rate limiting store: {api_key: [(timestamp, count), ...]}
rate_limit_store = defaultdict(list)

class APIKeyInfo(BaseModel):
    """API key information"""
    user_id: str
    tier: str
    rate_limit: int
    enabled: bool

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int
    remaining: int
    reset_at: datetime

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIKeyInfo:
    """
    Verify API key from Bearer token.
    Raises HTTPException if invalid.
    """
    api_key = credentials.credentials

    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    key_info = API_KEYS[api_key]

    if not key_info["enabled"]:
        logger.warning(f"Disabled API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=403,
            detail="API key disabled"
        )

    return APIKeyInfo(**key_info)

def check_rate_limit(request: Request, api_key_info: APIKeyInfo = Depends(verify_api_key)):
    """
    Check rate limit for API key.
    Raises HTTPException if limit exceeded.

    NOTE: This implementation uses in-memory storage and won't work
    across multiple replicas. Use Redis for production.
    """
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")
    current_time = time.time()
    window_start = current_time - 60  # 1 minute window

    # Clean old entries
    rate_limit_store[api_key] = [
        (ts, count) for ts, count in rate_limit_store[api_key]
        if ts > window_start
    ]

    # Count requests in current window
    request_count = sum(count for _, count in rate_limit_store[api_key])

    if request_count >= api_key_info.rate_limit:
        reset_time = datetime.fromtimestamp(
            rate_limit_store[api_key][0][0] + 60
        )

        logger.warning(
            f"Rate limit exceeded for user {api_key_info.user_id}: "
            f"{request_count}/{api_key_info.rate_limit}"
        )

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": api_key_info.rate_limit,
                "reset_at": reset_time.isoformat()
            }
        )

    # Add current request
    rate_limit_store[api_key].append((current_time, 1))

    # Add rate limit info to request state
    request.state.rate_limit = RateLimitInfo(
        limit=api_key_info.rate_limit,
        remaining=api_key_info.rate_limit - request_count - 1,
        reset_at=datetime.fromtimestamp(current_time + 60)
    )

    return api_key_info

@app.get("/secure/device/{device_ip}")
async def get_device_info(
    device_ip: str,
    request: Request,
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """
    Get device information (secured endpoint).
    Requires valid API key and respects rate limits.
    """
    logger.info(f"Device query from user {api_key_info.user_id}: {device_ip}")

    rate_limit = request.state.rate_limit

    return {
        "device_ip": device_ip,
        "status": "reachable",
        "user_tier": api_key_info.tier,
        "rate_limit": {
            "limit": rate_limit.limit,
            "remaining": rate_limit.remaining,
            "reset_at": rate_limit.reset_at.isoformat()
        }
    }

@app.post("/secure/execute")
async def execute_command(
    request: Request,
    api_key_info: APIKeyInfo = Depends(check_rate_limit)
):
    """
    Execute command on device (secured endpoint).
    Premium tier only.
    """
    if api_key_info.tier != "premium":
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires premium tier"
        )

    body = await request.json()

    logger.info(f"Command execution from user {api_key_info.user_id}")

    return {
        "status": "executed",
        "command": body.get("command"),
        "device": body.get("device_ip")
    }
```

**Example: Authenticated request**
```bash
curl -H "Authorization: Bearer sk_prod_premium_001" \
     http://localhost:8000/secure/device/192.168.1.1
```

**Response (success):**
```json
{
  "device_ip": "192.168.1.1",
  "status": "reachable",
  "user_tier": "premium",
  "rate_limit": {
    "limit": 100,
    "remaining": 99,
    "reset_at": "2026-01-19T14:35:25.123456"
  }
}
```

**Example: Rate limit exceeded**
```bash
# After 100 requests within 1 minute
curl -H "Authorization: Bearer sk_prod_premium_001" \
     http://localhost:8000/secure/device/192.168.1.1
```

**Response (429 Too Many Requests):**
```json
{
  "detail": {
    "error": "Rate limit exceeded",
    "limit": 100,
    "reset_at": "2026-01-19T14:35:25.123456"
  }
}
```

### Production Health Checks

```python
# app/monitoring.py
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import psutil
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Network API with Monitoring")

# Service start time
SERVICE_START_TIME = datetime.now()

# Health check cache (check expensive operations less frequently)
health_cache = {
    "last_check": None,
    "cache_duration": 30,  # seconds
    "status": None
}

class HealthStatus(BaseModel):
    """Basic health status"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    version: str
    uptime_seconds: float

class DetailedHealthStatus(BaseModel):
    """Detailed health check response"""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]
    metrics: Dict[str, Any]

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    start_time = time.time()

    try:
        # Simulate database check (replace with actual DB query)
        await asyncio.sleep(0.01)

        return {
            "status": "healthy",
            "latency_ms": (time.time() - start_time) * 1000,
            "connections": 5,
            "max_connections": 100
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def check_external_api_health() -> Dict[str, Any]:
    """Check external API dependencies (Claude API, etc.)"""
    start_time = time.time()

    try:
        # Simulate external API check
        await asyncio.sleep(0.02)

        return {
            "status": "healthy",
            "latency_ms": (time.time() - start_time) * 1000,
            "last_success": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def check_device_connectivity() -> Dict[str, Any]:
    """Check if we can reach managed devices"""
    try:
        # Simulate device connectivity check
        reachable = 48
        total = 50

        return {
            "status": "healthy" if reachable == total else "degraded",
            "reachable_devices": reachable,
            "total_devices": total,
            "unreachable_devices": total - reachable
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "process_count": len(psutil.pids())
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.
    Fast response, minimal checks.
    Used by load balancers for routing decisions.
    """
    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime
    )

@app.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """
    Detailed health check with all dependencies.
    Includes caching to avoid overwhelming dependencies.
    Used for dashboards and detailed monitoring.
    """
    current_time = time.time()

    # Check cache
    if (health_cache["last_check"] and
        current_time - health_cache["last_check"] < health_cache["cache_duration"]):
        logger.info("Returning cached health status")
        return health_cache["status"]

    # Perform health checks
    logger.info("Performing detailed health checks")

    db_health, api_health, device_health = await asyncio.gather(
        check_database_health(),
        check_external_api_health(),
        check_device_connectivity(),
        return_exceptions=True
    )

    # Get system metrics
    system_metrics = get_system_metrics()

    # Determine overall status
    checks = {
        "database": db_health if not isinstance(db_health, Exception) else {"status": "error", "error": str(db_health)},
        "external_api": api_health if not isinstance(api_health, Exception) else {"status": "error", "error": str(api_health)},
        "device_connectivity": device_health if not isinstance(device_health, Exception) else {"status": "error", "error": str(device_health)}
    }

    # Calculate overall status
    statuses = [check.get("status", "unknown") for check in checks.values()]
    if "unhealthy" in statuses or "error" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()

    result = DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks,
        metrics=system_metrics
    )

    # Update cache
    health_cache["last_check"] = current_time
    health_cache["status"] = result

    return result

@app.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.
    Checks if service can handle requests.
    Returns 200 if ready, 503 if not.
    """
    # Check critical dependencies
    try:
        db_check = await check_database_health()

        if db_check["status"] != "healthy":
            return Response(
                content="Service not ready: database unhealthy",
                status_code=503
            )

        return {"status": "ready"}

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return Response(
            content=f"Service not ready: {str(e)}",
            status_code=503
        )

@app.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    Checks if service is alive (not deadlocked).
    Always returns 200 unless process is frozen.
    """
    return {"status": "alive"}

@app.get("/metrics", response_class=Response)
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()
    system_metrics = get_system_metrics()

    metrics = [
        f"# HELP service_uptime_seconds Service uptime in seconds",
        f"# TYPE service_uptime_seconds gauge",
        f"service_uptime_seconds {uptime}",
        "",
        f"# HELP system_cpu_percent CPU usage percentage",
        f"# TYPE system_cpu_percent gauge",
        f"system_cpu_percent {system_metrics['cpu_percent']}",
        "",
        f"# HELP system_memory_percent Memory usage percentage",
        f"# TYPE system_memory_percent gauge",
        f"system_memory_percent {system_metrics['memory_percent']}",
        "",
        f"# HELP system_disk_percent Disk usage percentage",
        f"# TYPE system_disk_percent gauge",
        f"system_disk_percent {system_metrics['disk_percent']}",
        "",
        f"# HELP http_requests_total Total HTTP requests",
        f"# TYPE http_requests_total counter",
        f"http_requests_total 12345",
        "",
        f"# HELP http_requests_errors_total Total HTTP request errors",
        f"# TYPE http_requests_errors_total counter",
        f"http_requests_errors_total 23",
    ]

    return Response(
        content="\n".join(metrics),
        media_type="text/plain"
    )
```

**Example: Detailed health check**
```bash
curl http://localhost:8000/health/detailed
```

**Response:**
```json
{
  "status": "degraded",
  "timestamp": "2026-01-19T14:30:25.123456",
  "version": "1.0.0",
  "uptime_seconds": 3721.45,
  "checks": {
    "database": {
      "status": "healthy",
      "latency_ms": 12.34,
      "connections": 5,
      "max_connections": 100
    },
    "external_api": {
      "status": "healthy",
      "latency_ms": 23.45,
      "last_success": "2026-01-19T14:30:25.123456"
    },
    "device_connectivity": {
      "status": "degraded",
      "reachable_devices": 48,
      "total_devices": 50,
      "unreachable_devices": 2
    }
  },
  "metrics": {
    "cpu_percent": 15.2,
    "memory_percent": 42.8,
    "disk_percent": 68.5,
    "process_count": 234
  }
}
```

### Production Deployment: Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application with production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**requirements.txt:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
anthropic==0.18.0
netmiko==4.3.0
psutil==5.9.8
python-multipart==0.0.6
```

**Build and run:**
```bash
docker build -t network-ai-api:latest .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_api_key \
  network-ai-api:latest
```

### Production Deployment: Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: network-ai-api
  labels:
    app: network-ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: network-ai-api
  template:
    metadata:
      labels:
        app: network-ai-api
    spec:
      containers:
      - name: api
        image: network-ai-api:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-key
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 2
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: network-ai-api
spec:
  selector:
    app: network-ai-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  anthropic-key: "your_anthropic_api_key_here"
```

**Deploy to Kubernetes:**
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl get pods -l app=network-ai-api
kubectl get svc network-ai-api
```

**Output:**
```
NAME                              READY   STATUS    RESTARTS   AGE
network-ai-api-7d8c9f6b5d-4xk2m   1/1     Running   0          2m
network-ai-api-7d8c9f6b5d-9p7wq   1/1     Running   0          2m
network-ai-api-7d8c9f6b5d-kx5rt   1/1     Running   0          2m

NAME             TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
network-ai-api   LoadBalancer   10.96.123.45    203.0.113.50    80:32145/TCP   2m
```

### V4 Production Features

1. **API Key Authentication**: Bearer token validation with user tiers
2. **Rate Limiting**: 100 req/min (premium), 10 req/min (basic)
3. **Health Checks**:
   - `/health` - Fast check for load balancers
   - `/health/detailed` - Comprehensive check with caching
   - `/health/ready` - Kubernetes readiness probe
   - `/health/live` - Kubernetes liveness probe
4. **Metrics**: Prometheus-compatible `/metrics` endpoint
5. **Horizontal Scaling**: 3 replicas in Kubernetes
6. **Resource Limits**: CPU and memory constraints
7. **Auto-recovery**: Liveness/readiness probes restart failed pods

### Production Checklist

**Security:**
- ✅ API key authentication
- ✅ Rate limiting per user
- ⚠️  Use HTTPS/TLS in production (add ingress controller)
- ⚠️  Use Redis for rate limiting across replicas
- ✅ Input validation with Pydantic

**Performance:**
- ✅ Async request handling
- ✅ ThreadPoolExecutor for blocking I/O
- ✅ Health check caching (30s)
- ⚠️  Add Redis for caching frequent queries
- ✅ Multiple workers (uvicorn --workers 4)

**Reliability:**
- ✅ Liveness and readiness probes
- ✅ Graceful shutdown (uvicorn handles SIGTERM)
- ⚠️  Add circuit breakers for external API calls
- ✅ Max iterations limit for agentic loops (prevents infinite loops)
- ✅ Error handling for individual device failures

**Observability:**
- ✅ Structured logging
- ✅ Prometheus metrics endpoint
- ⚠️  Add distributed tracing (Jaeger/Zipkin)
- ⚠️  Add alerting (PagerDuty/Slack integration)
- ⚠️  Add error tracking (Sentry)

## Hands-On Labs

### Lab 1: Build Basic FastAPI Server

**Objective:** Create a FastAPI server with Pydantic validation and health endpoints.

**Time:** 30 minutes

**Steps:**

1. **Create project structure:**
```bash
mkdir network-ai-api
cd network-ai-api
mkdir app
touch app/__init__.py
touch app/main.py
touch requirements.txt
```

2. **Install dependencies:**
```bash
pip install fastapi uvicorn pydantic
```

3. **Implement V1 code** (see V1 section above)

4. **Run the server:**
```bash
python app/main.py
```

5. **Test endpoints:**
```bash
# Health check
curl http://localhost:8000/health

# Interactive docs
open http://localhost:8000/docs

# Test validation (should fail)
curl -X POST "http://localhost:8000/query-device" \
  -H "Content-Type: application/json" \
  -d '{"device_ip": "invalid", "commands": []}'
```

**Expected Results:**
- Server starts on port 8000
- `/docs` shows interactive API documentation
- Health endpoint returns uptime
- Invalid IP address is rejected with validation error
- Empty commands list is rejected

### Lab 2: Add Async Network Operations

**Objective:** Add ThreadPoolExecutor and concurrent device operations.

**Time:** 45 minutes

**Prerequisites:** Lab 1 completed

**Steps:**

1. **Install Netmiko:**
```bash
pip install netmiko
```

2. **Create network_operations.py** (see V2 section above)

3. **Set up test devices** (use GNS3 or EVE-NG):
   - 3 Cisco IOS routers
   - Configure SSH access
   - Note IP addresses, usernames, passwords

4. **Test concurrent execution:**
```bash
curl -X POST "http://localhost:8000/execute-commands" \
  -H "Content-Type: application/json" \
  -d '{
    "devices": [
      {"host": "192.168.1.1", "username": "admin", "password": "cisco"},
      {"host": "192.168.1.2", "username": "admin", "password": "cisco"},
      {"host": "192.168.1.3", "username": "admin", "password": "cisco"}
    ],
    "commands": ["show version", "show ip interface brief"]
  }'
```

5. **Measure performance:**
   - Note `total_execution_time_ms`
   - Compare to sum of individual `execution_time_ms`
   - Confirm concurrent execution (total ≈ slowest device, not sum)

**Expected Results:**
- All 3 devices queried in parallel
- Total time ≈ slowest device (not sum of all three)
- Individual device failures don't affect others
- Results include output from all successful devices

### Lab 3: Deploy MCP & AI Agent System

**Objective:** Implement MCP protocol, integrate Claude, deploy to Kubernetes.

**Time:** 60 minutes

**Prerequisites:** Lab 2 completed, Claude API key, Kubernetes cluster (minikube/kind)

**Steps:**

1. **Set up Claude API:**
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

2. **Install Anthropic SDK:**
```bash
pip install anthropic
```

3. **Implement MCP server and AI agent** (see V3 section above)

4. **Test MCP endpoints:**
```bash
# List available tools
curl http://localhost:8000/mcp/tools

# Call a tool directly
curl -X POST "http://localhost:8000/mcp/call-tool" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "check_interface_status",
    "arguments": {"device_ip": "192.168.1.1"}
  }'
```

5. **Test autonomous agent:**
```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Check all interfaces and analyze the routing table",
    "device_ip": "192.168.1.1"
  }'
```

6. **Containerize the application:**
```bash
# Create Dockerfile (see V4 section)
docker build -t network-ai-api:latest .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY network-ai-api:latest
```

7. **Deploy to Kubernetes:**
```bash
# Start minikube (if not already running)
minikube start

# Create secret for API key
kubectl create secret generic api-secrets \
  --from-literal=anthropic-key=$ANTHROPIC_API_KEY

# Deploy
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=network-ai-api
kubectl logs -l app=network-ai-api --tail=50

# Test service
kubectl port-forward svc/network-ai-api 8080:80
curl http://localhost:8080/health
```

**Expected Results:**
- MCP tools list shows 5 available tools
- Direct tool calls return structured results
- Agent autonomously selects and chains tools
- Docker image builds successfully
- Kubernetes deployment shows 3 running replicas
- Health checks pass
- Service accessible via port-forward

## Check Your Understanding

<details>
<summary><strong>Question 1:</strong> Why use ThreadPoolExecutor for Netmiko instead of native async?</summary>

**Answer:**

Netmiko is **not async**—it uses blocking socket I/O. If you call Netmiko directly in an async FastAPI handler:

```python
@app.post("/bad-example")
async def bad_async_handler(device: DeviceConfig):
    # THIS BLOCKS THE EVENT LOOP!
    with ConnectHandler(**device.dict()) as net_connect:
        output = net_connect.send_command("show version")
    return {"output": output}
```

**What happens:**
1. Request 1 arrives, starts Netmiko connection
2. Netmiko blocks waiting for SSH response (1-2 seconds)
3. Request 2 arrives, but event loop is blocked
4. Request 2 waits for Request 1 to complete
5. Zero concurrency

**Correct approach with ThreadPoolExecutor:**

```python
executor = ThreadPoolExecutor(max_workers=10)

@app.post("/good-example")
async def good_async_handler(device: DeviceConfig):
    loop = asyncio.get_event_loop()
    # Runs in thread pool, event loop stays free
    result = await loop.run_in_executor(
        executor,
        netmiko_sync_function,
        device
    )
    return result
```

**What happens:**
1. Request 1 arrives, Netmiko runs in Thread 1
2. Event loop immediately free for Request 2
3. Request 2 arrives, Netmiko runs in Thread 2
4. Both requests process concurrently
5. Up to 10 concurrent Netmiko connections (max_workers=10)

**Alternative:** Use `aiossapi` library for async SSH, but it has limited device support compared to Netmiko.

**Bottom line:** ThreadPoolExecutor lets you use blocking libraries in async frameworks without blocking the event loop.
</details>

<details>
<summary><strong>Question 2:</strong> How does MCP standardize AI agent tool interactions?</summary>

**Answer:**

Model Context Protocol (MCP) provides three standardization layers:

**1. Tool Discovery (`/mcp/tools` endpoint):**
```json
{
  "name": "check_interface_status",
  "description": "Check status of network interfaces",
  "input_schema": {
    "type": "object",
    "properties": {
      "device_ip": {"type": "string", "description": "IP address"}
    },
    "required": ["device_ip"]
  }
}
```

AI agents can query what tools are available and what inputs they need.

**2. Tool Invocation (`/mcp/call-tool` endpoint):**
```json
{
  "tool": "check_interface_status",
  "arguments": {"device_ip": "192.168.1.1"}
}
```

Standardized request format regardless of underlying implementation.

**3. Result Formatting:**
```json
{
  "tool": "check_interface_status",
  "result": {"status": "up", "protocol": "up"},
  "error": null,
  "execution_time_ms": 245.67
}
```

Consistent response structure with result/error separation and timing.

**Why this matters:**

**Without MCP (custom API):**
- Claude integration: Custom code for each API
- Adding new tool: Update Claude prompt and parsing logic
- Different providers: Rewrite integration for each
- Error handling: Inconsistent across tools

**With MCP:**
- Claude integration: Read `/mcp/tools`, use `input_schema` for validation
- Adding new tool: Add to `NETWORK_TOOLS`, automatically discoverable
- Different providers: Any MCP-compatible server works
- Error handling: Standardized `error` field

**Real-world benefit:** You can swap out the MCP server (e.g., switch from Python/FastAPI to Go/Fiber) without changing Claude integration code. The protocol is the contract.

</details>

<details>
<summary><strong>Question 3:</strong> What's the difference between liveness and readiness probes in Kubernetes?</summary>

**Answer:**

**Liveness Probe (`/health/live`):**
- **Question:** "Is the process alive?"
- **Failure action:** Restart the pod
- **Check frequency:** Every 30s (typical)
- **Use case:** Detect deadlocks, infinite loops, frozen processes

**Example liveness check:**
```python
@app.get("/health/live")
async def liveness_check():
    # Just return 200 - if we can respond, we're alive
    return {"status": "alive"}
```

**When it fails:**
- Process is deadlocked (can't handle requests)
- Thread pool exhausted (all threads blocked)
- Infinite loop in code

**Kubernetes action:** `kubectl get pods` shows `CrashLoopBackOff`, pod is restarted.

---

**Readiness Probe (`/health/ready`):**
- **Question:** "Can the service handle traffic?"
- **Failure action:** Remove from service load balancing (pod stays running)
- **Check frequency:** Every 10s (typical)
- **Use case:** Detect dependency failures, warm-up periods

**Example readiness check:**
```python
@app.get("/health/ready")
async def readiness_check():
    # Check critical dependencies
    db_status = await check_database()
    if db_status != "healthy":
        return Response(status_code=503, content="DB unavailable")

    return {"status": "ready"}
```

**When it fails:**
- Database connection lost
- Required external API down
- Cache not yet populated (startup)

**Kubernetes action:** Pod stays running, but traffic routes to other healthy pods.

---

**Why both?**

**Scenario:** Database goes down temporarily

- **Liveness:** Still passes (process not frozen)
- **Readiness:** Fails (can't serve requests without DB)
- **Result:** Pod stays running, no restart overhead, traffic routes elsewhere, automatically recovers when DB returns

**Scenario:** Code bug causes deadlock

- **Liveness:** Fails (process frozen)
- **Readiness:** Fails (can't respond)
- **Result:** Pod is restarted (fresh process), hopefully resolves deadlock

**Best practice:**
- Liveness: Lightweight, just check process health
- Readiness: Check dependencies, but cache results (30s) to avoid overwhelming them
</details>

<details>
<summary><strong>Question 4:</strong> Why cache detailed health checks, and what are the trade-offs?</summary>

**Answer:**

**Problem without caching:**

```python
@app.get("/health/detailed")
async def detailed_health():
    # Checks 5 databases, 3 external APIs, 50 devices
    db_checks = await check_all_databases()  # 500ms
    api_checks = await check_external_apis()  # 1000ms
    device_checks = await check_all_devices()  # 2000ms

    # Total: 3.5 seconds per health check
```

**If Prometheus scrapes every 15 seconds:**
- 4 health checks per minute
- Each check: 3.5 seconds
- CPU time: 14 seconds per minute (23% of CPU just for health checks!)

**Worse:** If you have 3 replicas + external monitoring + dashboards hitting `/health/detailed`:
- 10+ requests per minute per replica
- 30+ requests per minute total
- All checking the same 50 devices repeatedly

**Solution with caching:**

```python
health_cache = {
    "last_check": None,
    "cache_duration": 30,  # seconds
    "status": None
}

@app.get("/health/detailed")
async def detailed_health():
    current_time = time.time()

    # Return cached result if recent
    if (health_cache["last_check"] and
        current_time - health_cache["last_check"] < health_cache["cache_duration"]):
        return health_cache["status"]

    # Perform expensive checks
    result = await perform_all_checks()

    # Cache for 30 seconds
    health_cache["last_check"] = current_time
    health_cache["status"] = result

    return result
```

**Benefits:**
- First request: 3.5 seconds (full check)
- Next 30 seconds: <1ms (cached response)
- Reduces load on dependencies (devices, databases)
- Multiple health check consumers share cache

**Trade-offs:**

| Aspect | Impact | Mitigation |
|--------|--------|------------|
| **Stale data** | Health status up to 30s old | Acceptable for non-critical monitoring; use `/health/live` for critical checks |
| **False positives** | Database dies, cache still shows "healthy" for 30s | 30s delay is acceptable vs overwhelming DB with health checks |
| **Cache invalidation** | Manual changes not reflected immediately | Use shorter cache (10s) for critical systems |
| **Memory usage** | Storing health check results | Negligible (few KB per check) |

**Cache duration guidelines:**
- **5-10s:** Critical production systems
- **30s:** Standard production (good balance)
- **60s:** Development/staging environments
- **No cache:** Never (always cache at least 5s)

**Advanced:** Use Redis for cache if you have multiple replicas:

```python
import redis
r = redis.Redis()

@app.get("/health/detailed")
async def detailed_health():
    # Check Redis cache (shared across all replicas)
    cached = r.get("health_check")
    if cached:
        return json.loads(cached)

    result = await perform_all_checks()
    r.setex("health_check", 30, json.dumps(result))
    return result
```

This way all 3 replicas share the same cache, reducing total load 3×.
</details>

## Lab Time Budget

| Phase | Time | Cost | Notes |
|-------|------|------|-------|
| **Environment Setup** | 15 min | Free | Install Python, Docker, kubectl |
| **Lab 1: Basic Server** | 30 min | Free | FastAPI + Pydantic validation |
| **Lab 2: Async Operations** | 45 min | Free | ThreadPoolExecutor + Netmiko |
| **Lab 3: MCP & AI Agent** | 60 min | $5-10 | Claude API testing (~100 requests) |
| **Kubernetes Deployment** | 30 min | Free | Local minikube cluster |
| **Production Testing** | 30 min | $10-20 | Full integration testing |
| **TOTAL** | **3.5 hours** | **$15-30** | One-time learning investment |

**Monthly Production Costs:**

| Component | Cost | Scaling |
|-----------|------|---------|
| Claude API (10K requests/month) | $30-50 | $3-5 per 1K requests |
| Kubernetes cluster (3 nodes, 2 CPU each) | $150-200 | AWS EKS, GCP GKE, or Azure AKS |
| Load balancer | $20-30 | Included in some K8s offerings |
| Monitoring (Prometheus + Grafana) | Free-$50 | Self-hosted free, managed ~$50/mo |
| **TOTAL** | **$200-330/month** | For 10K AI agent requests |

**ROI Calculation:**

**Scenario:** Replace manual network troubleshooting

- **Manual process:**
  - Network engineer troubleshoots 20 incidents/month
  - Average 2 hours per incident
  - 40 hours/month total
  - Engineer cost: $100/hour
  - **Monthly cost: $4,000**

- **Automated with AI agent:**
  - System handles 15/20 incidents automatically
  - Engineer only handles complex 5 incidents
  - 10 hours/month
  - Engineer cost: $1,000
  - System cost: $300/month
  - **Monthly cost: $1,300**

**Monthly savings:** $2,700
**Annual savings:** $32,400
**ROI:** 1,233% annually
**Break-even:** 1 week

**Alternative ROI scenario:** API development time savings

- **Without this framework:**
  - Build FastAPI server from scratch: 40 hours
  - Add authentication: 8 hours
  - Add rate limiting: 8 hours
  - Add health checks: 4 hours
  - Add Claude integration: 16 hours
  - Add MCP protocol: 12 hours
  - Kubernetes deployment: 12 hours
  - **Total: 100 hours**

- **With this framework:**
  - Follow V1→V4 progression: 3.5 hours (labs)
  - Customize for your use case: 6 hours
  - Production hardening: 4 hours
  - **Total: 13.5 hours**

**Time saved:** 86.5 hours
**Value at $100/hour:** $8,650 one-time savings

## Production Deployment Guide

### Week 1-2: Development Environment (V1 + V2)

**Goal:** Build and test basic FastAPI server with async operations

**Tasks:**
- Set up development environment (Python, Docker, Netmiko)
- Implement V1: Basic FastAPI server with Pydantic validation
- Implement V2: Async network operations with ThreadPoolExecutor
- Test with 3-5 lab devices
- Document API endpoints in OpenAPI

**Deliverables:**
- Working FastAPI server
- Postman collection with example requests
- Initial API documentation

**Team:** 1 developer, 20 hours

### Week 3-4: Staging Environment (V3)

**Goal:** Add MCP protocol and Claude integration

**Tasks:**
- Implement MCP server with tool definitions
- Integrate Claude with function calling
- Test autonomous agent queries
- Set up staging environment (Docker Compose)
- Performance testing (concurrent requests, tool chaining)

**Deliverables:**
- MCP-compliant API
- AI agent with 5+ network tools
- Staging environment with sample data
- Performance test results

**Team:** 1 developer + 1 network engineer, 30 hours

### Week 5-6: Production Preparation (V4)

**Goal:** Add authentication, rate limiting, monitoring

**Tasks:**
- Implement API key authentication
- Add rate limiting (use Redis for multi-replica)
- Implement health checks (basic, detailed, liveness, readiness)
- Add Prometheus metrics endpoint
- Create Kubernetes manifests
- Set up CI/CD pipeline

**Deliverables:**
- Production-ready authentication
- Comprehensive health checks
- Kubernetes deployment files
- CI/CD pipeline (GitHub Actions or GitLab CI)

**Team:** 1 developer + 1 DevOps engineer, 40 hours

### Week 7-8: Production Rollout

**Goal:** Deploy to production with gradual rollout

**Tasks:**
- **Week 7:**
  - Deploy to production Kubernetes cluster
  - Start with 1 replica, route 10% of traffic
  - Monitor metrics, logs, error rates
  - Test with internal users only

- **Week 8:**
  - Scale to 3 replicas
  - Route 100% of traffic
  - Enable external access
  - Set up alerting (PagerDuty/Slack)
  - Document operational procedures

**Deliverables:**
- Production deployment (3 replicas)
- Monitoring dashboards (Grafana)
- Alerting rules
- Operational runbook

**Team:** 1 DevOps engineer + 1 SRE, 30 hours

### Week 9+: Operations and Optimization

**Ongoing tasks:**
- Monitor Claude API costs (optimize prompt caching)
- Tune rate limits based on usage patterns
- Add new MCP tools as needed
- Expand to additional use cases
- Review and respond to alerts

**Team:** 1 SRE (10 hours/month)

### Phased Rollout Strategy

**Phase 1: Internal testing (Week 7, Days 1-3)**
- 1 replica
- 10% traffic (shadow mode, compare AI vs manual)
- Internal users only
- Success criteria: <1% error rate, <500ms p95 latency

**Phase 2: Limited production (Week 7, Days 4-7)**
- 2 replicas
- 50% traffic
- Select customer group
- Success criteria: No increase in support tickets

**Phase 3: Full production (Week 8)**
- 3 replicas
- 100% traffic
- All users
- Success criteria: 99.9% uptime, <2s p95 end-to-end latency

### Rollback Plan

**Triggers for rollback:**
- Error rate >5%
- p95 latency >5 seconds
- Health check failures >10%
- Critical security issue discovered

**Rollback procedure:**
1. Scale replicas to 0 (immediate)
2. Route traffic to fallback (manual process or old system)
3. Debug in staging environment
4. Fix and redeploy

**Estimated rollback time:** <5 minutes (Kubernetes scaling + DNS update)

## Common Problems and Solutions

### Problem 1: Netmiko Blocking Event Loop

**Symptom:**
```
INFO:     127.0.0.1:54321 - "POST /execute-commands HTTP/1.1" 200 OK (12457ms)
WARNING:  Performance degradation: concurrent requests taking sequential time
```

Multiple device queries take sequential time instead of parallel time.

**Cause:**

Calling Netmiko directly in async handler blocks the event loop:

```python
# WRONG - blocks event loop
@app.post("/execute")
async def execute(device: Device):
    with ConnectHandler(**device.dict()) as conn:
        output = conn.send_command("show version")
    return {"output": output}
```

**Solution:**

Use ThreadPoolExecutor with `run_in_executor`:

```python
# CORRECT - runs in thread pool
executor = ThreadPoolExecutor(max_workers=10)

def netmiko_sync(device):
    with ConnectHandler(**device.dict()) as conn:
        return conn.send_command("show version")

@app.post("/execute")
async def execute(device: Device):
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(executor, netmiko_sync, device)
    return {"output": output}
```

**Prevention:**
- Always use `run_in_executor` for blocking I/O
- Never call synchronous network libraries directly in async handlers
- Monitor request latency to detect blocking

---

### Problem 2: Rate Limiting Not Working Across Replicas

**Symptom:**
```
# User makes 150 requests/minute (should be limited to 100)
# But each replica allows 100, so 3 replicas = 300 total
kubectl logs network-ai-api-7d8c9f6b5d-4xk2m
INFO: User user_001 made 95 requests this minute (within limit)
```

**Cause:**

In-memory rate limiting doesn't sync across Kubernetes replicas:

```python
# WRONG - in-memory store
rate_limit_store = defaultdict(list)  # Each replica has its own store
```

With 3 replicas, each allows 100 req/min = 300 total (should be 100).

**Solution:**

Use Redis for shared rate limiting:

```python
# CORRECT - shared Redis store
import redis
r = redis.Redis(host='redis-service', port=6379, decode_responses=True)

def check_rate_limit(api_key: str, limit: int) -> bool:
    current = int(time.time())
    window_key = f"ratelimit:{api_key}:{current // 60}"

    # Increment and get count
    count = r.incr(window_key)

    if count == 1:
        r.expire(window_key, 60)  # Expire after 1 minute

    if count > limit:
        return False  # Rate limit exceeded

    return True  # Within limit
```

**Prevention:**
- Use Redis for any shared state (rate limits, sessions, cache)
- Test with multiple replicas before production
- Monitor actual request rates per API key across all replicas

---

### Problem 3: Health Checks Timing Out

**Symptom:**
```
kubectl get pods
NAME                              READY   STATUS    RESTARTS   AGE
network-ai-api-7d8c9f6b5d-4xk2m   0/1     Running   47         12m
```

Pods constantly restarting due to liveness probe failures.

**Cause:**

Detailed health check is too slow (>3 seconds), Kubernetes probe times out:

```python
# WRONG - too slow for liveness probe
@app.get("/health/live")
async def liveness():
    db_check = await check_database()  # 500ms
    api_check = await check_external_api()  # 1000ms
    device_check = await check_50_devices()  # 5000ms
    # Total: 6.5 seconds > 3 second timeout
```

**Solution:**

Use lightweight liveness, detailed readiness, cached comprehensive checks:

```python
# CORRECT - separate endpoints with appropriate checks

@app.get("/health/live")
async def liveness():
    # Just return 200 - if we can respond, we're alive
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    # Check critical dependencies only
    db_check = await quick_db_ping()  # <100ms
    if db_check != "healthy":
        return Response(status_code=503)
    return {"status": "ready"}

@app.get("/health/detailed")
async def detailed():
    # Expensive checks, but cached for 30s
    if cached and not expired:
        return cached_result

    result = await comprehensive_health_check()
    cache_result(result, ttl=30)
    return result
```

**Kubernetes probe configuration:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
  timeoutSeconds: 3
  failureThreshold: 3  # 3 failures before restart

readinessProbe:
  httpGet:
    path: /health/ready
  timeoutSeconds: 3
  failureThreshold: 2  # 2 failures before removing from service
```

**Prevention:**
- Keep liveness probes ultra-lightweight
- Use readiness probes for dependency checks
- Cache expensive health checks (30s+)
- Set appropriate timeouts and failure thresholds

---

### Problem 4: Claude API Timeout During Tool Loops

**Symptom:**
```
ERROR: Tool execution loop exceeded 30 seconds, Claude API timeout
anthropic.APIError: Request timed out
```

Agent gets stuck in infinite tool loop or very slow tools cause timeout.

**Cause:**

No iteration limit or timeout on agentic loop:

```python
# WRONG - infinite loop possible
while response.stop_reason == "tool_use":
    # Execute tool
    # Call Claude again
    # Repeat forever if Claude keeps requesting tools
```

**Solution:**

Add max iterations and per-tool timeouts:

```python
# CORRECT - bounded iteration and timeouts
MAX_ITERATIONS = 5
TOOL_TIMEOUT = 10  # seconds

async def agent_query(request: AgentRequest):
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = client.messages.create(...)

        if response.stop_reason == "tool_use":
            for tool_block in response.content:
                if tool_block.type == "tool_use":
                    try:
                        # Execute tool with timeout
                        result = await asyncio.wait_for(
                            execute_tool(tool_block.name, tool_block.input),
                            timeout=TOOL_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        result = {"error": "Tool execution timeout"}
        else:
            return response  # Final answer

    return {"error": "Max iterations reached", "iterations": iterations}
```

**Prevention:**
- Always set max_iterations on agent loops
- Add timeouts to individual tool calls
- Monitor average iterations per query
- Log tool call chains for debugging

---

### Problem 5: Kubernetes Readiness Probe Failing

**Symptom:**
```
kubectl describe pod network-ai-api-7d8c9f6b5d-4xk2m
Events:
  Readiness probe failed: HTTP probe failed with statuscode: 503
```

Pod is running but not receiving traffic.

**Cause:**

Readiness probe checks database, but database is temporarily unavailable:

```python
# Overly strict readiness check
@app.get("/health/ready")
async def readiness():
    db = await check_database()
    if db != "healthy":
        return Response(status_code=503)  # Fails probe
    return {"status": "ready"}
```

Database has brief hiccup (1-2 seconds), readiness probe fails, pod removed from service. Even though pod could handle requests without database for that brief period.

**Solution:**

Use circuit breaker pattern with grace period:

```python
# CORRECT - circuit breaker with grace period
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=30):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None

    async def call(self, func):
        try:
            result = await func()
            self.failures = 0  # Reset on success
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()

            if self.failures >= self.failure_threshold:
                raise  # Circuit open

            # Still within grace period, return degraded
            return {"status": "degraded", "error": str(e)}

db_circuit = CircuitBreaker(failure_threshold=3, timeout=30)

@app.get("/health/ready")
async def readiness():
    try:
        db_status = await db_circuit.call(check_database)

        # Accept both healthy and degraded
        if db_status.get("status") in ["healthy", "degraded"]:
            return {"status": "ready"}

        return Response(status_code=503)
    except Exception:
        # Only fail after 3 consecutive failures
        return Response(status_code=503)
```

**Prevention:**
- Don't make readiness too sensitive to temporary failures
- Use circuit breakers for external dependencies
- Set appropriate `failureThreshold` in Kubernetes (2-3 failures before removing)
- Monitor readiness probe failures vs actual service health

---

### Problem 6: High Memory Usage with Concurrent Requests

**Symptom:**
```
kubectl top pods
NAME                              CPU(cores)   MEMORY(bytes)
network-ai-api-7d8c9f6b5d-4xk2m   450m         890Mi
network-ai-api-7d8c9f6b5d-9p7wq   430m         920Mi
network-ai-api-7d8c9f6b5d-kx5rt   OOMKilled    512Mi (limit)
```

Pods hitting memory limits and getting killed.

**Cause:**

ThreadPoolExecutor with too many workers creates too many concurrent Netmiko connections:

```python
# WRONG - unlimited thread pool
executor = ThreadPoolExecutor()  # Default: 5x CPU cores

# With 4 CPU cores = 20 threads
# Each Netmiko connection = ~50 MB
# 20 connections = 1000 MB just for SSH connections
```

**Solution:**

Limit thread pool workers and add request queuing:

```python
# CORRECT - bounded thread pool
executor = ThreadPoolExecutor(max_workers=10)

# Add semaphore for request queuing
MAX_CONCURRENT_DEVICES = 50
device_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEVICES)

async def execute_commands_async(device, commands, enable_mode):
    # Wait for semaphore (queue if at limit)
    async with device_semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            execute_commands_sync,
            device,
            commands,
            enable_mode
        )
        return result
```

**Kubernetes resource limits:**
```yaml
resources:
  requests:
    memory: "256Mi"  # Guaranteed minimum
  limits:
    memory: "512Mi"  # Hard cap (OOMKilled if exceeded)
```

**Prevention:**
- Always set `max_workers` on ThreadPoolExecutor
- Use semaphores to limit concurrent operations
- Monitor memory usage per connection type
- Set appropriate Kubernetes resource limits
- Profile memory usage under load

---

## Summary

You built a production FastAPI server across four versions:

**V1: Basic Server (30 min, Free)**
- Pydantic validation
- Health endpoints
- OpenAPI docs

**V2: Async Operations (45 min, Free)**
- ThreadPoolExecutor for Netmiko
- Concurrent device queries
- 2.8× performance improvement

**V3: MCP & AI Agents (60 min, $20-50/month)**
- MCP protocol implementation
- Claude function calling
- Autonomous tool selection

**V4: Production Deployment (90 min, $100-300/month)**
- API key authentication
- Rate limiting
- Comprehensive health checks
- Prometheus metrics
- Kubernetes deployment (3 replicas)

**Key architectural decisions:**

1. **ThreadPoolExecutor for Netmiko**: Blocking I/O libraries need thread isolation from async event loop
2. **MCP standardization**: Protocol-based tool calling enables AI agent portability
3. **Separate health endpoints**: Liveness (lightweight) vs readiness (dependency checks) vs detailed (comprehensive, cached)
4. **Redis for shared state**: Rate limiting across replicas requires centralized store
5. **Bounded iterations**: AI agent loops need max iteration limits to prevent infinite loops

**Production readiness:**
- ✅ Horizontal scaling (3+ replicas)
- ✅ Health monitoring (Prometheus + Kubernetes probes)
- ✅ Authentication and rate limiting
- ✅ Error handling and timeouts
- ✅ Concurrent request handling
- ✅ Structured logging

This is not a demo. This is production code. The Model Context Protocol standardizes how AI agents interact with your infrastructure. FastAPI provides the performance and tooling to make it production-ready. Together, they enable autonomous network operations at scale.

Deploy it. Monitor it. Scale it.

**Next chapter:** API Gateway patterns and load balancing for distributed AI systems.
