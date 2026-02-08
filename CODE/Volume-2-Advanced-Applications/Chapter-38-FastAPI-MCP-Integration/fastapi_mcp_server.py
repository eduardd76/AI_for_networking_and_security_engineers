"""
Chapter 38: FastAPI Server with MCP Integration
Production REST API Gateway with MCP Tool Bridges

This server provides REST API endpoints that bridge to MCP (Model Context Protocol)
tools for network device operations. Complete with auth, rate limiting, and monitoring.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Header, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import uvicorn

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Network Operations API with MCP",
    description="Production-ready REST API with MCP tool integration",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# In-memory storage (use Redis in production)
api_keys_db = {
    "sk_prod_12345": {"user": "admin", "tenant": "acme_corp", "tier": "premium"},
    "sk_dev_67890": {"user": "developer", "tenant": "test_org", "tier": "free"}
}

rate_limits = defaultdict(list)
request_logs = []

# MCP Tools Registry (simulated - in production, connect to real MCP server)
MCP_TOOLS = {
    "get_device_status": {
        "description": "Get operational status of a network device",
        "parameters": {"hostname": "string"},
        "example": {"hostname": "router-01"}
    },
    "get_interface_stats": {
        "description": "Get interface statistics",
        "parameters": {"hostname": "string", "interface": "string"},
        "example": {"hostname": "router-01", "interface": "Gi0/1"}
    },
    "run_show_command": {
        "description": "Execute show command on device",
        "parameters": {"hostname": "string", "command": "string"},
        "example": {"hostname": "switch-01", "command": "show ip interface brief"}
    },
    "check_bgp_status": {
        "description": "Check BGP neighbor status",
        "parameters": {"hostname": "string"},
        "example": {"hostname": "router-01"}
    }
}


# Pydantic Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or question")
    tools: Optional[List[str]] = Field(default=None, description="MCP tools to use")
    stream: Optional[bool] = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    response: str
    tools_used: List[str]
    execution_time_ms: float
    timestamp: datetime


class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolExecutionResponse(BaseModel):
    tool_name: str
    result: Any
    execution_time_ms: float
    success: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    mcp_tools_available: int
    uptime_seconds: float
    timestamp: datetime


class UsageStats(BaseModel):
    requests_today: int
    rate_limit_status: str
    tier: str


# Auth Middleware
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify API key and return user info"""
    api_key = credentials.credentials

    if api_key not in api_keys_db:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    user_info = api_keys_db[api_key]

    # Log request
    request_logs.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_info["user"],
        "tenant": user_info["tenant"]
    })

    return user_info


# Rate Limiting
async def check_rate_limit(request: Request, user_info: Dict = Depends(verify_api_key)):
    """Check rate limits based on tier"""
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")
    tier = user_info["tier"]

    # Rate limits by tier
    limits = {
        "free": {"requests": 10, "window": 60},
        "premium": {"requests": 100, "window": 60}
    }

    limit_config = limits.get(tier, limits["free"])
    window_start = datetime.now() - timedelta(seconds=limit_config["window"])

    # Clean old entries
    rate_limits[api_key] = [
        ts for ts in rate_limits[api_key]
        if ts > window_start
    ]

    # Check limit
    if len(rate_limits[api_key]) >= limit_config["requests"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit_config['requests']} requests per {limit_config['window']}s"
        )

    # Add current request
    rate_limits[api_key].append(datetime.now())


# MCP Tool Execution (simulated)
async def execute_mcp_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute MCP tool - in production, this would call actual MCP server
    For demo, we simulate the tool execution
    """
    await asyncio.sleep(0.5)  # Simulate network latency

    if tool_name == "get_device_status":
        return {
            "hostname": parameters.get("hostname"),
            "status": "online",
            "uptime": "45 days, 3 hours",
            "cpu_usage": 23,
            "memory_usage": 45
        }

    elif tool_name == "get_interface_stats":
        return {
            "hostname": parameters.get("hostname"),
            "interface": parameters.get("interface"),
            "status": "up",
            "speed": "1000Mbps",
            "input_packets": 1234567,
            "output_packets": 987654,
            "errors": 0
        }

    elif tool_name == "run_show_command":
        return {
            "hostname": parameters.get("hostname"),
            "command": parameters.get("command"),
            "output": "Interface GigabitEthernet0/1\\n  IP address: 10.0.0.1\\n  Status: up\\n  Protocol: up"
        }

    elif tool_name == "check_bgp_status":
        return {
            "hostname": parameters.get("hostname"),
            "bgp_neighbors": [
                {"neighbor": "10.1.1.1", "state": "Established", "uptime": "2d4h"},
                {"neighbor": "10.2.2.2", "state": "Established", "uptime": "5d12h"}
            ],
            "total_prefixes": 150000
        }

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown tool: {tool_name}"
        )


# Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Network Operations API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (no auth required)"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        mcp_tools_available=len(MCP_TOOLS),
        uptime_seconds=123456.78,
        timestamp=datetime.now()
    )


@app.get("/tools")
async def list_tools(user_info: Dict = Depends(verify_api_key)):
    """List available MCP tools"""
    return {
        "tools": MCP_TOOLS,
        "count": len(MCP_TOOLS),
        "user": user_info["user"]
    }


@app.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request: ToolExecutionRequest,
    user_info: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    Execute a single MCP tool

    Example:
    ```
    POST /tools/execute
    Authorization: Bearer sk_prod_12345
    {
        "tool_name": "get_device_status",
        "parameters": {"hostname": "router-01"}
    }
    ```
    """
    start_time = datetime.now()

    if request.tool_name not in MCP_TOOLS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request.tool_name}' not found"
        )

    try:
        result = await execute_mcp_tool(request.tool_name, request.parameters)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolExecutionResponse(
            tool_name=request.tool_name,
            result=result,
            execution_time_ms=execution_time,
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_info: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    Chat endpoint with automatic MCP tool selection

    Example:
    ```
    POST /chat
    Authorization: Bearer sk_prod_12345
    {
        "message": "Check the status of router-01",
        "tools": ["get_device_status", "check_bgp_status"]
    }
    ```
    """
    start_time = datetime.now()

    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0
    )

    # If tools specified, execute them first
    tool_results = []
    tools_used = []

    if request.tools:
        for tool_name in request.tools:
            if tool_name in MCP_TOOLS:
                # Extract parameters from message (simplified - in production use better parsing)
                parameters = {}
                if "router-01" in request.message.lower():
                    parameters["hostname"] = "router-01"
                elif "switch-01" in request.message.lower():
                    parameters["hostname"] = "switch-01"

                result = await execute_mcp_tool(tool_name, parameters)
                tool_results.append({
                    "tool": tool_name,
                    "result": result
                })
                tools_used.append(tool_name)

    # Build prompt with tool results
    prompt = f"User question: {request.message}\n\n"

    if tool_results:
        prompt += "Tool execution results:\n"
        for tr in tool_results:
            prompt += f"\n{tr['tool']}:\n{json.dumps(tr['result'], indent=2)}\n"
        prompt += "\nProvide a concise answer based on the tool results above."
    else:
        prompt += "Provide a helpful response."

    # Call LLM
    response = llm.invoke(prompt)

    execution_time = (datetime.now() - start_time).total_seconds() * 1000

    return ChatResponse(
        response=response.content,
        tools_used=tools_used,
        execution_time_ms=execution_time,
        timestamp=datetime.now()
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_info: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    Streaming chat endpoint

    Example:
    ```
    POST /chat/stream
    Authorization: Bearer sk_prod_12345
    {
        "message": "Explain BGP route selection",
        "stream": true
    }
    ```
    """
    async def generate():
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            streaming=True
        )

        async for chunk in llm.astream(request.message):
            yield f"data: {json.dumps({'content': chunk.content})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/stats", response_model=UsageStats)
async def get_usage_stats(user_info: Dict = Depends(verify_api_key)):
    """Get usage statistics for current API key"""
    # Count requests from this user today
    today = datetime.now().date()
    requests_today = sum(
        1 for log in request_logs
        if log["user"] == user_info["user"] and
        datetime.fromisoformat(log["timestamp"]).date() == today
    )

    return UsageStats(
        requests_today=requests_today,
        rate_limit_status="healthy",
        tier=user_info["tier"]
    )


# Example client code
def example_client():
    """Example of calling the API"""
    import requests

    BASE_URL = "http://localhost:8000"
    API_KEY = "sk_prod_12345"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Example 1: List available tools
    print("Example 1: List MCP Tools")
    response = requests.get(f"{BASE_URL}/tools", headers=headers)
    print(json.dumps(response.json(), indent=2))
    print()

    # Example 2: Execute a tool
    print("Example 2: Execute Tool")
    tool_request = {
        "tool_name": "get_device_status",
        "parameters": {"hostname": "router-01"}
    }
    response = requests.post(
        f"{BASE_URL}/tools/execute",
        json=tool_request,
        headers=headers
    )
    print(json.dumps(response.json(), indent=2))
    print()

    # Example 3: Chat with tool execution
    print("Example 3: Chat with Tools")
    chat_request = {
        "message": "Check the status and BGP neighbors of router-01",
        "tools": ["get_device_status", "check_bgp_status"]
    }
    response = requests.post(
        f"{BASE_URL}/chat",
        json=chat_request,
        headers=headers
    )
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("🚀 Starting FastAPI Server with MCP Integration")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /health                - Health check (no auth)")
    print("  GET  /tools                 - List MCP tools")
    print("  POST /tools/execute         - Execute single tool")
    print("  POST /chat                  - Chat with tool execution")
    print("  POST /chat/stream           - Streaming chat")
    print("  GET  /stats                 - Usage statistics")
    print()
    print("API Keys (for demo):")
    print("  sk_prod_12345  - Premium tier (100 req/min)")
    print("  sk_dev_67890   - Free tier (10 req/min)")
    print()
    print("MCP Tools Available:")
    for tool_name in MCP_TOOLS:
        print(f"  - {tool_name}")
    print()
    print("Documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
