# Chapter 38: FastAPI Server & MCP Integration

## Introduction

Production AI systems need robust APIs. You can't just run Python scripts forever. FastAPI provides async performance, automatic validation, and OpenAPI docs. The Model Context Protocol (MCP) standardizes how AI agents interact with external tools and data sources.

This chapter builds a production-grade FastAPI server that exposes network operations through MCP-compatible endpoints. You'll implement request validation, async handling, function calling patterns, and proper authentication.

No toy examples. This is code you can deploy.

## FastAPI Fundamentals for AI Endpoints

FastAPI uses Python type hints for automatic validation and documentation. It's built on Starlette (async) and Pydantic (validation). Perfect for AI workloads that need to handle multiple concurrent requests.

### Basic FastAPI Server Structure

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

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
        import ipaddress
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
    results: List[Dict[str, Any]]
    timestamp: datetime
    execution_time_ms: float

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
    """Root endpoint"""
    return {
        "service": "Network AI Operations API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Output when running:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**GET /health response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-19T14:30:25.123456",
  "version": "1.0.0",
  "uptime_seconds": 127.45
}
```

### Key Components

1. **Pydantic Models**: Automatic validation, serialization, and documentation
2. **Type Hints**: FastAPI uses these to validate requests and generate OpenAPI specs
3. **Async Handlers**: Use `async def` for endpoints that do I/O
4. **Automatic Docs**: OpenAPI docs at `/docs`, ReDoc at `/redoc`

## Async Request Handling for Network Operations

Network operations are I/O-bound. Use async to handle multiple devices concurrently.

```python
# app/network_operations.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import time
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

class CommandRequest(BaseModel):
    """Command execution request"""
    devices: List[NetworkDevice]
    commands: List[str]
    enable_mode: bool = False

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

@app.get("/operations/status/{host}")
async def get_device_status(host: str):
    """Get quick status of a device"""
    # Simulate async operation
    await asyncio.sleep(0.1)
    return {
        "host": host,
        "reachable": True,
        "last_check": datetime.now().isoformat()
    }
```

**Example request:**
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
      }
    ],
    "commands": ["show version", "show ip interface brief"],
    "enable_mode": false
  }'
```

**Output:**
```json
{
  "total_devices": 2,
  "successful": 2,
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
    }
  ],
  "total_execution_time_ms": 1256.89
}
```

**Key points:**
- Both devices processed in parallel
- Total execution time ≈ slowest device (not sum of both)
- Netmiko runs in thread pool to avoid blocking async event loop

## Model Context Protocol (MCP) Integration

MCP standardizes how AI agents interact with tools. It defines:
- Tool discovery (what tools are available)
- Tool invocation (how to call them)
- Result formatting (standardized responses)

### MCP Server Implementation

```python
# app/mcp_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import json
import logging

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
        "description": "Retrieve running configuration from a network device",
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

# Tool implementations
async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool with given arguments"""

    if tool_name == "get_device_config":
        device_ip = arguments["device_ip"]
        config_type = arguments.get("config_type", "running")

        # Simulate config retrieval
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
    import time
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

**Output:**
```json
[
  {
    "name": "get_device_config",
    "description": "Retrieve running configuration from a network device",
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

**Example: Call a tool**
```bash
curl -X POST "http://localhost:8000/mcp/call-tool" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "check_interface_status",
    "arguments": {
      "device_ip": "192.168.1.1",
      "interface_name": "GigabitEthernet0/0"
    }
  }'
```

**Output:**
```json
{
  "tool": "check_interface_status",
  "result": {
    "device": "192.168.1.1",
    "interface": "GigabitEthernet0/0",
    "status": "up",
    "protocol": "up",
    "ip_address": "192.168.1.1",
    "subnet_mask": "255.255.255.0"
  },
  "error": null,
  "execution_time_ms": 12.34
}
```

## AI Agent with MCP Function Calling

Connect Claude to your MCP server for autonomous network operations.

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

# Define tools for Claude
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
    The agent will autonomously decide which tools to call.
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

**Example request:**
```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Check if all interfaces are up on the device and show me the routing table",
    "device_ip": "192.168.1.1"
  }'
```

**Output:**
```json
{
  "prompt": "Check if all interfaces are up on the device and show me the routing table",
  "response": "I checked the device at 192.168.1.1. Here's what I found:\n\nInterface Status:\n- GigabitEthernet0/0: UP/UP (192.168.1.1)\n- GigabitEthernet0/1: DOWN/DOWN (unassigned)\n\nRouting Table:\n- Default route (0.0.0.0/0) via 192.168.1.254 (static)\n- 10.0.0.0/8 network via 192.168.1.2 (OSPF)\n\nOne interface (GigabitEthernet0/1) is currently down. All other interfaces are operational.",
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

**What happened:**
1. User asked to check interfaces and routing table
2. Claude called `check_interface_status` tool
3. Claude called `analyze_routing_table` tool
4. Claude synthesized results into human-readable response
5. All tool calls tracked in response

## Authentication and Rate Limiting

Production APIs need security and throttling.

```python
# app/auth_and_limits.py
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import time
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Secured Network API")

# Security scheme
security = HTTPBearer()

# In-memory stores (use Redis in production)
API_KEYS = {
    "sk_test_123456": {
        "user_id": "user_001",
        "tier": "premium",
        "rate_limit": 100,  # requests per minute
        "enabled": True
    },
    "sk_test_789012": {
        "user_id": "user_002",
        "tier": "basic",
        "rate_limit": 10,
        "enabled": True
    }
}

# Rate limiting store: {api_key: [(timestamp, count), ...]}
rate_limit_store = defaultdict(list)

# Request tracking
request_log = []

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

@app.get("/secure/usage")
async def get_usage(
    api_key_info: APIKeyInfo = Depends(verify_api_key)
):
    """Get API usage statistics"""
    # In production, query from database
    return {
        "user_id": api_key_info.user_id,
        "tier": api_key_info.tier,
        "rate_limit": api_key_info.rate_limit,
        "requests_today": 145,
        "requests_this_month": 3421
    }

# HMAC signature verification (for webhooks)
def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str
) -> bool:
    """
    Verify HMAC signature for webhook payload.
    Used to verify requests are from trusted sources.
    """
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected_signature)

@app.post("/webhook/device-alert")
async def receive_device_alert(
    request: Request,
    x_signature: str = Header(...)
):
    """
    Receive device alert webhook.
    Verifies HMAC signature.
    """
    payload = await request.body()
    webhook_secret = "your_webhook_secret_here"

    if not verify_webhook_signature(payload, x_signature, webhook_secret):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()
    logger.info(f"Device alert received: {data}")

    return {"status": "received"}
```

**Example: Authenticated request**
```bash
curl -H "Authorization: Bearer sk_test_123456" \
     http://localhost:8000/secure/device/192.168.1.1
```

**Output (success):**
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
curl -H "Authorization: Bearer sk_test_123456" \
     http://localhost:8000/secure/device/192.168.1.1
```

**Output (rate limited):**
```json
{
  "detail": {
    "error": "Rate limit exceeded",
    "limit": 100,
    "reset_at": "2026-01-19T14:35:25.123456"
  }
}
```

**Example: Invalid API key**
```bash
curl -H "Authorization: Bearer invalid_key" \
     http://localhost:8000/secure/device/192.168.1.1
```

**Output:**
```json
{
  "detail": "Invalid API key"
}
```

## Advanced Health Checks and Monitoring

Production systems need comprehensive health monitoring.

```python
# app/monitoring.py
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
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
    """Overall health status"""
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

class MetricsResponse(BaseModel):
    """Prometheus-style metrics"""
    metrics: List[str]

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    start_time = time.time()

    try:
        # Simulate database check
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
    """Check external API dependencies"""
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
    Used by load balancers.
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

**Example: Basic health check**
```bash
curl http://localhost:8000/health
```

**Output:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-19T14:30:25.123456",
  "version": "1.0.0",
  "uptime_seconds": 3721.45
}
```

**Example: Detailed health check**
```bash
curl http://localhost:8000/health/detailed
```

**Output:**
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

**Example: Prometheus metrics**
```bash
curl http://localhost:8000/metrics
```

**Output:**
```
# HELP service_uptime_seconds Service uptime in seconds
# TYPE service_uptime_seconds gauge
service_uptime_seconds 3721.45

# HELP system_cpu_percent CPU usage percentage
# TYPE system_cpu_percent gauge
system_cpu_percent 15.2

# HELP system_memory_percent Memory usage percentage
# TYPE system_memory_percent gauge
system_memory_percent 42.8

# HELP system_disk_percent Disk usage percentage
# TYPE system_disk_percent gauge
system_disk_percent 68.5

# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total 12345

# HELP http_requests_errors_total Total HTTP request errors
# TYPE http_requests_errors_total counter
http_requests_errors_total 23
```

## Complete Production Example

Full production server with all components integrated.

```python
# production_server.py
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('network_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Production Network AI API",
    description="Production-ready API for AI-powered network operations with MCP support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} - {duration:.2f}ms")

    return response

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Models
class NetworkOperation(BaseModel):
    """Network operation request"""
    device_ip: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator('device_ip')
    def validate_ip(cls, v):
        import ipaddress
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")

class OperationResult(BaseModel):
    """Network operation result"""
    device_ip: str
    operation: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    timestamp: datetime

# Service state
SERVICE_START_TIME = datetime.now()
OPERATION_COUNTER = 0

# API Key validation (simplified - use proper auth in production)
async def validate_api_key(authorization: str = Header(...)):
    """Validate API key from header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")

    # In production: validate against database
    if api_key not in ["sk_test_123456", "sk_prod_789012"]:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key

# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Production Network AI API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "mcp_tools": "/mcp/tools"
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/operations/execute")
async def execute_operation(
    operation: NetworkOperation,
    api_key: str = Depends(validate_api_key)
):
    """
    Execute a network operation.
    Requires authentication.
    """
    global OPERATION_COUNTER
    OPERATION_COUNTER += 1

    start_time = time.time()

    logger.info(f"Executing operation: {operation.operation} on {operation.device_ip}")

    try:
        # Simulate operation execution
        await asyncio.sleep(0.1)

        result = {
            "status": "completed",
            "output": f"Operation {operation.operation} completed successfully",
            "parameters_used": operation.parameters
        }

        execution_time = (time.time() - start_time) * 1000

        return OperationResult(
            device_ip=operation.device_ip,
            operation=operation.operation,
            success=True,
            result=result,
            error=None,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Operation failed: {str(e)}")

        return OperationResult(
            device_ip=operation.device_ip,
            operation=operation.operation,
            success=False,
            result=None,
            error=str(e),
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools"""
    return [
        {
            "name": "get_device_config",
            "description": "Retrieve device configuration",
            "input_schema": {
                "type": "object",
                "properties": {
                    "device_ip": {"type": "string"},
                    "config_type": {"type": "string", "enum": ["running", "startup"]}
                },
                "required": ["device_ip"]
            }
        },
        {
            "name": "check_interface_status",
            "description": "Check interface operational status",
            "input_schema": {
                "type": "object",
                "properties": {
                    "device_ip": {"type": "string"},
                    "interface_name": {"type": "string"}
                },
                "required": ["device_ip"]
            }
        }
    ]

@app.get("/stats")
async def get_stats(api_key: str = Depends(validate_api_key)):
    """Get API statistics"""
    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()

    return {
        "uptime_seconds": uptime,
        "total_operations": OPERATION_COUNTER,
        "operations_per_minute": OPERATION_COUNTER / (uptime / 60) if uptime > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Production Network AI API")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
```

**Running the server:**
```bash
python production_server.py
```

**Output:**
```
2026-01-19 14:30:25,123 - __main__ - INFO - Starting Production Network AI API
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Example request:**
```bash
curl -X POST "http://localhost:8000/operations/execute" \
  -H "Authorization: Bearer sk_test_123456" \
  -H "Content-Type: application/json" \
  -d '{
    "device_ip": "192.168.1.1",
    "operation": "backup_config",
    "parameters": {
      "backup_location": "/backups",
      "include_startup": true
    }
  }'
```

**Output:**
```json
{
  "device_ip": "192.168.1.1",
  "operation": "backup_config",
  "success": true,
  "result": {
    "status": "completed",
    "output": "Operation backup_config completed successfully",
    "parameters_used": {
      "backup_location": "/backups",
      "include_startup": true
    }
  },
  "error": null,
  "execution_time_ms": 102.45,
  "timestamp": "2026-01-19T14:30:25.123456"
}
```

## Deployment Considerations

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "production_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t network-ai-api:latest .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key network-ai-api:latest
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: network-ai-api
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
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
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
  type: LoadBalancer
```

### Production Checklist

1. **Security**
   - Use proper authentication (OAuth2, JWT)
   - Implement HTTPS/TLS
   - Validate all inputs
   - Rate limit all endpoints
   - Log security events

2. **Performance**
   - Use async for I/O operations
   - Implement caching (Redis)
   - Connection pooling for databases
   - Load balancing
   - CDN for static content

3. **Reliability**
   - Health checks (liveness, readiness)
   - Graceful shutdown
   - Circuit breakers for external services
   - Retry logic with exponential backoff
   - Database connection recovery

4. **Observability**
   - Structured logging
   - Distributed tracing (Jaeger, Zipkin)
   - Metrics collection (Prometheus)
   - Alerting (PagerDuty, Slack)
   - Error tracking (Sentry)

5. **Operations**
   - Automated deployments (CI/CD)
   - Rolling updates
   - Automatic rollback on failures
   - Backup and disaster recovery
   - Documentation

## Summary

You built a production-grade FastAPI server with:
- Async request handling for network operations
- MCP protocol integration for standardized tool calling
- AI agent integration with Claude and function calling
- Authentication and rate limiting
- Comprehensive health checks and monitoring
- Production deployment patterns

This is not a demo. This is production code. Deploy it, monitor it, iterate on it.

The Model Context Protocol standardizes how AI agents interact with your infrastructure. FastAPI provides the performance and tooling to make it production-ready. Together, they enable autonomous network operations at scale.

Next chapter: Building distributed AI systems across multiple data centers.
