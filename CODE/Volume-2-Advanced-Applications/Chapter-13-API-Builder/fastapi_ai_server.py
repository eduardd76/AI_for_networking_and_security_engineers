"""
Chapter 13: Building AI-Powered REST APIs
FastAPI Server with LLM Integration for Network Operations

This script demonstrates building a production-ready REST API that integrates
AI for network device analysis, with authentication, rate limiting, and monitoring.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
from collections import defaultdict
from langchain_anthropic import ChatAnthropic
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Network AI API",
    description="AI-powered network device analysis API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Rate limiting (in-memory for demo, use Redis in production)
rate_limit_data = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10

# API Keys (in production, use a database)
VALID_API_KEYS = {
    "demo-key-123": {"name": "Demo User", "tier": "free"},
    "prod-key-456": {"name": "Production User", "tier": "premium"}
}

# LLM client
llm = None


# Pydantic Models
class DeviceConfig(BaseModel):
    hostname: str
    config: str
    device_type: Optional[str] = "cisco_ios"


class AnalysisRequest(BaseModel):
    config: str
    analysis_type: str = Field(default="security", description="Type: security, optimization, compliance")


class AnalysisResponse(BaseModel):
    analysis: str
    recommendations: List[str]
    severity: str
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    timestamp: datetime


class LogEntry(BaseModel):
    log_text: str
    context: Optional[str] = None


class LogAnalysisResponse(BaseModel):
    severity: str
    summary: str
    root_cause: Optional[str]
    suggested_actions: List[str]


# Dependency: Verify API Key
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify API key from Bearer token"""
    api_key = credentials.credentials

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return VALID_API_KEYS[api_key]


# Dependency: Rate Limiting
async def check_rate_limit(api_key: str = Header(...)):
    """Check rate limit for API key"""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    rate_limit_data[api_key] = [
        timestamp for timestamp in rate_limit_data[api_key]
        if timestamp > window_start
    ]

    # Check limit
    if len(rate_limit_data[api_key]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW}s"
        )

    # Add current request
    rate_limit_data[api_key].append(now)


# Initialize LLM
@app.on_event("startup")
async def startup_event():
    """Initialize LLM client on startup"""
    global llm
    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print("✅ LLM client initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize LLM: {e}")


# Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_available=llm is not None,
        timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (no auth required)"""
    return HealthResponse(
        status="healthy" if llm else "degraded",
        version="1.0.0",
        llm_available=llm is not None,
        timestamp=datetime.now()
    )


@app.post("/api/v1/analyze/config", response_model=AnalysisResponse)
async def analyze_config(
    request: AnalysisRequest,
    user: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    Analyze network device configuration using AI

    Example:
    ```
    POST /api/v1/analyze/config
    Authorization: Bearer demo-key-123
    {
        "config": "interface GigabitEthernet0/1\\n ip address 10.0.0.1 255.255.255.0",
        "analysis_type": "security"
    }
    ```
    """
    if not llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service unavailable"
        )

    try:
        # Build prompt based on analysis type
        if request.analysis_type == "security":
            prompt = f"""Analyze this network configuration for security issues:

{request.config}

Provide:
1. Security issues found (be specific)
2. Severity assessment (Critical/High/Medium/Low)
3. Remediation recommendations

Format as concise bullet points."""

        elif request.analysis_type == "optimization":
            prompt = f"""Analyze this network configuration for optimization opportunities:

{request.config}

Provide:
1. Performance optimizations
2. Best practices not followed
3. Configuration improvements

Format as concise bullet points."""

        elif request.analysis_type == "compliance":
            prompt = f"""Check this network configuration against industry best practices:

{request.config}

Provide:
1. Compliance gaps
2. Missing security controls
3. Required changes

Format as concise bullet points."""

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analysis_type: {request.analysis_type}"
            )

        # Call LLM
        response = llm.invoke(prompt)

        # Parse response
        analysis_text = response.content
        recommendations = [
            line.strip().lstrip("- ")
            for line in analysis_text.split("\n")
            if line.strip() and line.strip().startswith("-")
        ][:5]  # Top 5 recommendations

        # Determine severity from analysis
        severity = "Medium"
        if any(word in analysis_text.lower() for word in ["critical", "severe", "urgent"]):
            severity = "Critical"
        elif any(word in analysis_text.lower() for word in ["high", "important", "significant"]):
            severity = "High"
        elif any(word in analysis_text.lower() for word in ["low", "minor", "informational"]):
            severity = "Low"

        return AnalysisResponse(
            analysis=analysis_text,
            recommendations=recommendations,
            severity=severity,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze/logs", response_model=LogAnalysisResponse)
async def analyze_logs(
    log_entry: LogEntry,
    user: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    Analyze network syslog messages using AI

    Example:
    ```
    POST /api/v1/analyze/logs
    Authorization: Bearer demo-key-123
    {
        "log_text": "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down",
        "context": "Production core router"
    }
    ```
    """
    if not llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service unavailable"
        )

    try:
        context_note = f"\nContext: {log_entry.context}" if log_entry.context else ""

        prompt = f"""Analyze this network syslog message:

{log_entry.log_text}{context_note}

Provide:
1. Severity (Critical/High/Medium/Low)
2. Brief summary (one sentence)
3. Most likely root cause
4. 2-3 specific troubleshooting actions

Be concise and actionable."""

        response = llm.invoke(prompt)
        analysis = response.content

        # Parse severity
        severity = "Medium"
        if "critical" in analysis.lower():
            severity = "Critical"
        elif "high" in analysis.lower():
            severity = "High"
        elif "low" in analysis.lower():
            severity = "Low"

        # Extract summary (first line)
        lines = [l.strip() for l in analysis.split("\n") if l.strip()]
        summary = lines[0] if lines else "Log analysis completed"

        # Extract root cause
        root_cause = None
        for line in lines:
            if "root cause" in line.lower() or "likely cause" in line.lower():
                root_cause = line

        # Extract actions
        actions = []
        for line in lines:
            if line.startswith("- ") or line.startswith("• "):
                actions.append(line.lstrip("- •").strip())

        return LogAnalysisResponse(
            severity=severity,
            summary=summary,
            root_cause=root_cause,
            suggested_actions=actions[:3] if actions else ["Review device logs", "Check connectivity", "Verify configuration"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Log analysis failed: {str(e)}"
        )


@app.get("/api/v1/devices")
async def list_devices(
    user: Dict = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """
    List network devices (demo data)

    Example:
    ```
    GET /api/v1/devices
    Authorization: Bearer demo-key-123
    ```
    """
    # Mock data
    devices = [
        {"id": 1, "hostname": "core-rtr-01", "type": "router", "status": "active"},
        {"id": 2, "hostname": "edge-sw-01", "type": "switch", "status": "active"},
        {"id": 3, "hostname": "dist-sw-02", "type": "switch", "status": "maintenance"},
    ]

    return {
        "devices": devices,
        "count": len(devices),
        "user": user["name"],
        "tier": user["tier"]
    }


@app.get("/api/v1/stats")
async def get_stats(user: Dict = Depends(verify_api_key)):
    """
    Get API usage statistics

    Example:
    ```
    GET /api/v1/stats
    Authorization: Bearer demo-key-123
    ```
    """
    # In production, track this in a database
    return {
        "user": user["name"],
        "tier": user["tier"],
        "requests_today": 42,  # Mock
        "rate_limit": f"{RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
        "endpoints_available": 6,
        "llm_status": "available" if llm else "unavailable"
    }


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


# Example client code
def example_client():
    """Example of calling the API from Python"""
    import requests

    BASE_URL = "http://localhost:8000"
    API_KEY = "demo-key-123"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Example 1: Health check
    print("Example 1: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(response.json())
    print()

    # Example 2: Analyze config
    print("Example 2: Analyze Configuration")
    config_request = {
        "config": "line vty 0 4\n transport input telnet\n password cisco123",
        "analysis_type": "security"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/analyze/config",
        json=config_request,
        headers=headers
    )
    print(response.json())
    print()

    # Example 3: Analyze log
    print("Example 3: Analyze Syslog")
    log_request = {
        "log_text": "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired",
        "context": "Core router in DC1"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/analyze/logs",
        json=log_request,
        headers=headers
    )
    print(response.json())


if __name__ == "__main__":
    print("🚀 Starting Network AI API Server")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /health                    - Health check (no auth)")
    print("  GET  /api/v1/devices            - List devices")
    print("  POST /api/v1/analyze/config     - Analyze configuration")
    print("  POST /api/v1/analyze/logs       - Analyze syslog")
    print("  GET  /api/v1/stats              - Usage statistics")
    print()
    print("API Keys (for demo):")
    print("  demo-key-123  - Free tier (10 req/min)")
    print("  prod-key-456  - Premium tier (10 req/min)")
    print()
    print("Documentation: http://localhost:8000/docs")
    print("=" * 60)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
